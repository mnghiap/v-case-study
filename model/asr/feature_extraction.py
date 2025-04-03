import torch
import torch.nn as nn
from torchaudio.transforms import MFCC

class AudioFeatureExtraction(nn.Module):
    """
    Simple MFCC feature extraction
    """
    def __init__(self, mfcc_config):
        super().__init__()
        self.mfcc = MFCC(**mfcc_config)

    def forward(self, raw_wave):
        """
        Perform MFCC feature extraction on audio raw wave

        Args:
            raw_wave: (B, n_audio_samples)

        Returns:
            features: (B, T, n_features)
        """
        with torch.no_grad():
            features = self.mfcc(raw_wave) # (B, n_features, T)
        features = features.transpose(1, 2)
        return features
