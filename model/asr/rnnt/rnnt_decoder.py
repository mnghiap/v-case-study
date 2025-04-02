import torch
import torch.nn as nn

from model.asr.rnnt.rnnt import RNNT
from model.lm.lstm_lm import LSTM_LM
from model.vocab import Vocab

LOG_ZERO = -1e25

class RNNT_Decoder:
    """
    RNNT decoder with external LM.
    Comes with a simple greedy search (beam search with beam size 1).
    This actually uses the RNA topology for decoding, which is simpler to implement.
    There's also a functionaility to keep rescoring with entities.

    Args:
        rnnt: RNNT model
        lm: external LM used in decoding
        vocab: Vocab object
    """
    def __init__(self, rnnt: RNNT, lm: LSTM_LM, vocab: Vocab):
        self.rnnt = rnnt
        self.lm = lm
        self.vocab = vocab
    
    
    @torch.no_grad()
    def greedy_search(self, raw_wave, predictor_scale=1.0, lm_scale=0.0, force_entity_constraint=False):
        """
        Perform a simple greedy search by taking maximum label emission at each time frame.
        Taking predictor output into account. Uses external LM.

        Equivalent to a beam search with beam size 1.

        This is actually RNA decoding, not RNNT, but we do this for simplicity.

        Args:
            raw_wave: raw audio wave (B, T)
            predictor_scale: Control the predictor influence in the logits per time step
            lm_scale: LM weight in decoding
            force_entity_constraint: Force the constraint that after emitting the
                <ENTITY> symbol, the decoder must emit an entity from the list
                and then another <ENTITY>

        Returns:

        """
        self.rnnt.eval()
        self.lm.eval()
        features = self.rnnt.feature_extraction(raw_wave)
        encoder_output = self.rnnt.encoder(features)
        batch_size, max_input_time, _ = encoder_output.shape

        # store hypotheses
        hyp_w_blank = [[] for _ in range(batch_size)] # store emitted labels (including blank) per time frame
        hyp_no_blank = [[] for _ in range(batch_size)] # store emitted true labels (no blank)

        # prepare predictor state
        last_predictor_symbol = self.rnnt.predictor.get_initial_predictor_symbol(batch_size) # always (B, 1)
        predictor_state = self.rnnt.predictor.get_initial_state(batch_size) # always (n_lstm_layers, B, hidden_dim)

        # prepare lm_state
        use_lm = lm_scale > 0.0
        if use_lm:
            last_lm_symbol = self.lm.get_bos_symbol(batch_size)
            lm_state = self.lm.get_initial_state(batch_size)

        if force_entity_constraint:
            # mask to tell which indices are entities, blank or is entity boundary
            is_entity = self.vocab.generate_entity_mask() # (V+1)
            is_entity_BV = is_entity.unsqueeze(0).expand(batch_size, -1) # (B, V+1)
            is_entity_boundary = torch.full((self.vocab.vocab_size_no_blank+1,), fill_value=False, dtype=torch.long) # (V+1)
            is_entity_boundary[self.vocab.entity_boundary_idx] = True
            is_entity_boundary_BV = is_entity_boundary.unsqueeze(0).expand(batch_size, -1) # (B, V+1)

            # prepare a mask to tell which batch must emit an entity or another <ENTITY> token
            must_emit_entity_boundary = torch.full((batch_size,), fill_value=False, dtype=torch.bool) # (B,)
            must_emit_entity = torch.full((batch_size,), fill_value=False, dtype=torch.bool) # (B,)

        for t in range(max_input_time):
            # Each "step" here are the logits over V + blank
            encoder_step = encoder_output[:, t:t+1, :].squeeze(1) # (B, V+1)
            predictor_step, predictor_state_new = self.rnnt.predictor.step(last_predictor_symbol, predictor_state)
            predictor_step = predictor_step.squeeze(1) # (B, V+1)

            # compute rnnt output for current time step
            logits = self.rnnt.joiner.step(encoder_step, predictor_scale*predictor_step) # (B, V+1)
            logits = logits.log_softmax(-1)

            # compute LM score and add to the AM score
            if use_lm:
                lm_step, lm_state_new = self.lm.step(last_lm_symbol, lm_state) # (B, 1, V), no blank
                lm_step = lm_step.squeeze(1).log_softmax(-1) # (B, V)
                # recombination
                logits[:, :-1] += lm_step * lm_scale # because blank is last

            # mask out SB, since it's only used for the predictor and LM initial state
            logits[:, self.vocab.sb_idx] = LOG_ZERO

            # if forcing entity constraint, check if the model must emit entity of entity boundary
            # and then mask out the corresponding probability
            if force_entity_constraint:
                must_emit_entity_BV = must_emit_entity.unsqueeze(-1).expand(-1, self.vocab.vocab_size_no_blank+1)
                logits = torch.where(
                    torch.logical_and(must_emit_entity_BV, torch.logical_not(is_entity_BV)),
                    LOG_ZERO,
                    logits
                )
                must_emit_entity_boundary_BV = must_emit_entity_boundary.unsqueeze(-1).expand(-1, self.vocab.vocab_size_no_blank+1) # (B, V+1)
                logits = torch.where(
                    torch.logical_and(must_emit_entity_boundary_BV, torch.logical_not(is_entity_boundary_BV)),
                    LOG_ZERO,
                    logits
                )
            # Now take the best output, append to the hypotheses
            best_beam, best_beam_idx = torch.topk(logits, k=1, dim=-1) # both are (B, 1)
            for b, best_output in enumerate(best_beam_idx.squeeze(1).tolist()): # (B,)
                hyp_w_blank[b].append(best_output)
                if best_output != self.vocab.blank_idx:
                    hyp_no_blank[b].append(best_output)
            
            # update the next predictor and LM symbol and states if a non-blank symbol is emitted
            emitted_symbol = torch.tensor([hyp[-1] for hyp in hyp_w_blank], dtype=torch.long)
            not_emitted_blank = emitted_symbol != self.vocab.blank_idx # (B, )

            # update lm state
            if use_lm:
                last_lm_symbol = torch.where(not_emitted_blank, emitted_symbol, last_lm_symbol)
                update_lm_state = not_emitted_blank.unsqueeze(0).unsqueeze(-1).expand(self.lm.n_lstm_layers, -1, self.lm.hidden_dim)
                lm_h0, lm_c0 = lm_state
                lm_hn, lm_cn = lm_state_new
                lm_hn = torch.where(update_lm_state, lm_hn, lm_h0)
                lm_cn = torch.where(update_lm_state, lm_cn, lm_c0)
                lm_state = (lm_hn, lm_cn)

            # update predictor state
            last_predictor_symbol = torch.where(not_emitted_blank, emitted_symbol, last_predictor_symbol)
            update_predictor_state = not_emitted_blank.unsqueeze(0).unsqueeze(-1).expand(
                self.rnnt.predictor.n_lstm_layers,
                -1,
                self.rnnt.predictor.hidden_dim
            )
            predictor_h0, predictor_c0 = predictor_state
            predictor_hn, predictor_cn = predictor_state_new
            predictor_hn = torch.where(update_predictor_state, predictor_hn, predictor_h0)
            predictor_cn = torch.where(update_predictor_state, predictor_cn, predictor_c0)
            predictor_state = (predictor_hn, predictor_cn)

            # update the maskings for entity constraints
            # this code only works because an entity is just one token
            if force_entity_constraint:
                must_emit_entity_or_blank_old = must_emit_entity.clone()
                must_emit_entity = torch.logical_and(
                    emitted_symbol == self.vocab.entity_boundary_idx,
                    torch.logical_not(must_emit_entity_boundary)
                )
                
                must_emit_entity_boundary = torch.logical_and(
                    must_emit_entity_or_blank_old,
                    torch.isin(emitted_symbol, torch.tensor(self.vocab.entity_idxs, dtype=torch.long))
                )

        return hyp_no_blank
