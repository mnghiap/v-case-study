import torch
import torch.nn as nn
from torchaudio.functional import rnnt_loss

from model.asr.rnnt.rnnt import RNNT
from model.lm.lstm_lm import LSTM_LM
from model.vocab import Vocab 

class EntityListRescorer:
    """
    This rescorer replaces some tokens in the hypotheses with
    actual entities and then do rescoring.
    """
    def __init__(self, rnnt: RNNT, lm: LSTM_LM, vocab: Vocab):
        self.rnnt = rnnt
        self.lm = lm
        self.vocab = vocab


    def scorer(self, raw_wave, hyps, lm_scale=0.0):
        """
        Scoring hypotheses with RNNT loss and LM score.

        Args:
            raw_wave: (B, T)
            hyps: (B, S)

        Returns:
         scores: log p(hyp | audio) + lm_scale*log p(hyp) (B,)
        """
        logits = self.rnnt(raw_wave, hyps) # (B, T, S+1, V+1)
        rnnt_score = rnnt_loss( # -log p(hyp | audio), #(B,)
            logits=logits,
            targets=hyps,
            logit_lengths=torch.full((logits.shape[0],), fill_value=logits.shape[1]),
            target_lengths=torch.full((logits.shape[0],), fill_value=hyps.shape[1]),
            blank=logits.shape[-1] - 1,
            reduction="none",
            fused_log_softmax=True,
            )
        lm_score = 0.0
        if lm_scale > 0.0:
            lm_score = self.lm.compute_lm_score(hyps) # log p(hyp) # (B,)
        return -rnnt_score + lm_score * lm_scale


    def rescore_with_allowed_entities(self, raw_wave, hyp, entity_pos, beam_size):
        """
        Given a single hypothesis hyp (for each batch) with the entity boundary tokens,
        replace the token inside the entity boundaries with actual entity
        names in the stored entity list, and try to find the best hyp.

        Behaves similar to a label-synchronous beam search (over the 
        marked positions, vocabulary is the list of entities).
        """
        

