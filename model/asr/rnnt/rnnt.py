import torch
import torch.nn as nn

from model.asr.feature_extraction import AudioFeatureExtraction
from model.asr.rnnt.encoder import Encoder
from model.asr.rnnt.predictor import Predictor
from model.asr.rnnt.joiner import Joiner
from model.lm.lstm_lm import LSTM_LM

class RNNT(nn.Module):
    """
    RNNT model like the original paper with encoder, predictor, and joiner networks

    Important constraint: blank must always be the last in the output dim!

    Args:
        feature_extraction_config:
        encoder_config:
        predictor_config:
        joiner_config:
    """
    def __init__(self, feature_extraction_config, encoder_config, predictor_config, joiner_config):
        super().__init__()
        self.feature_extraction = AudioFeatureExtraction(feature_extraction_config)
        self.encoder = Encoder(**encoder_config)
        self.predictor = Predictor(**predictor_config)
        self.joiner = Joiner(**joiner_config)


    def forward(self, raw_wave, target):
        features = self.feature_extraction(raw_wave)
        encoder_output = self.encoder(features)
        predictor_output, _ = self.predictor(target)
        logits = self.joiner(encoder_output, predictor_output) # (B, T, S+1, V+1)
        return logits

        