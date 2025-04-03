import torch
import torch.nn as nn
from torchaudio.functional import rnnt_loss

from model.asr.rnnt.rnnt import RNNT
from model.lm.lstm_lm import LSTM_LM
from model.vocab import Vocab 
from model.entity_recognizer.blstm_entity_recognizer import BLSTM_Entity_Recognizer

class EntityRescorer:
    """
    This rescorer replaces some tokens in the hypotheses with
    actual entities and then do rescoring.
    """
    def __init__(self, rnnt: RNNT, lm: LSTM_LM, vocab: Vocab, entity_recognizer: BLSTM_Entity_Recognizer):
        self.rnnt = rnnt
        self.lm = lm
        self.vocab = vocab
        self.entity_recognizer = entity_recognizer


    def scorer(self, raw_wave, hyps, lm_scale=0.0):
        """
        Scoring hypotheses with RNNT loss and LM score.

        Args:
            raw_wave: (B, n_audio_samples)
            hyps: (B, S)

        Returns:
         scores: log p(hyp | audio) + lm_scale*log p(hyp) (B,)
        """
        logits = self.rnnt(raw_wave, hyps) # (B, T, S+1, V+1)
        rnnt_score = rnnt_loss( # -log p(hyp | audio), #(B,)
            logits=logits,
            targets=hyps.int(),
            logit_lengths=torch.full((logits.shape[0],), fill_value=logits.shape[1], dtype=torch.int),
            target_lengths=torch.full((logits.shape[0],), fill_value=hyps.shape[1], dtype=torch.int),
            blank=logits.shape[-1] - 1,
            reduction="none",
            fused_log_softmax=True,
            )
        lm_score = 0.0
        if lm_scale > 0.0:
            lm_score = self.lm.compute_lm_score(hyps) # log p(hyp) # (B,)
        return -rnnt_score + lm_score * lm_scale


    @torch.no_grad()
    def rescore_with_allowed_entities(self, raw_wave, hyp, beam_size, lm_scale, rescore_valid_entities=True):
        """
        Given a single hypothesis hyp (for each batch) with the entity boundary tokens,
        replace the token inside the entity boundaries with actual entity
        names in the stored entity list, and try to find the best hyp.

        Behaves similar to a label-synchronous beam search (over the 
        marked positions, vocabulary is the list of entities).

        Even though there is a batch dim in the inputs, this function would only
        work for batch size 1.

        Args:
            raw_wave: (1, n_audio_samples)
            hyp: (1, S)
            entity_pos: (1, S)
            beam_size: how many entities to consider
            lm_scale:
            rescore_valid_entities: For some tokens recognized as an entity, the token
                itself might already be in the list of allowed entities. If True, then
                still reconsider othre allowed entities for such positions. If False,
                leave these positions as they are.

        Returns:
            hyps_beams (Tensor): hypothesized rescored sequences of shape (1, beam_size)
        """
        n_entities = len(self.vocab.entity_idxs)
        assert beam_size <= n_entities, \
            "Beam size must be no larger than the size of the list of entities"
        assert raw_wave.shape[0] == 1 and hyp.shape[0] == 1, \
            "This rescoring only accept batch size 1, with 1 raw wave and 1 single hypothesis"
        torch_entity_list = torch.tensor(self.vocab.entity_idxs).long().unsqueeze(0).unsqueeze(0).expand(1, beam_size, n_entities) # (B, beam, n_entities)
        
        # compute the position where there is entity
        is_entity_log_prob = self.entity_recognizer(hyp).log_softmax(-1) # (B, S, 2)
        entity_pos = torch.argmax(is_entity_log_prob, dim=-1) # (B, S)
        entity_pos_idxs = entity_pos.squeeze(0).nonzero().squeeze(-1) # (masked positions,)
        
        # prepare for rescoring
        raw_wave_beams = raw_wave.expand(beam_size*n_entities, -1)
        max_target_len = hyp.shape[1]
        
        # hypotheses
        hyps_beams = hyp.unsqueeze(1).expand(-1, beam_size, -1) # (B, beam, S)
        first_beam = True

        # beam search over masked positions
        for pos in entity_pos_idxs.tolist():
            if not rescore_valid_entities and hyp[0][pos] in self.vocab.entity_idxs:
                continue
            hyps_beams_extended = hyps_beams.unsqueeze(2).expand(-1, -1, n_entities, -1).clone() # (B, beam, n_entities, S)
            hyps_beams_extended[:, :, :, pos] = torch_entity_list
            hyps_scores = self.scorer(
                raw_wave_beams,
                hyps_beams_extended.squeeze(0).view(beam_size*n_entities, max_target_len),
                lm_scale=lm_scale
            )
            hyps_scores = hyps_scores.view(1, beam_size, n_entities)
            if first_beam: # if first beam, only look at the first beam_size scores
                hyps_scores[:, 1:, :] = -float("inf")
                first_beam = False
            hyps_scores = hyps_scores.view(1, beam_size*n_entities)
            top_beam_scores, top_beam_idx = torch.topk(hyps_scores, k=beam_size, dim=-1)
            hyps_beams = hyps_beams_extended.view(1, beam_size*n_entities, max_target_len).gather(1, top_beam_idx.unsqueeze(-1).expand(-1, -1, max_target_len))
        
        return hyps_beams
