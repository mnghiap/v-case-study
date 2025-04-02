import torch
import torch.nn as nn

# for typing hint
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

    def rescore_with_allowed_entities(self, raw_wave, hyp, entity_pos, beam_size=1):
        """
        Given a single hypothesis hyp (for each batch) with the entity boundary tokens,
        replace the token inside the entity boundaries with actual entity
        names in the stored entity list, and try to find the best hyp.

        Behaves similar to a label-synchronous beam search (over the 
        marked positions, vocabulary is the list of entities).
        """
        

