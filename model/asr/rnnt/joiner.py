import torch
import torch.nn as nn

class Joiner(nn.Module):
    """
    Simple sum joiner like the original RNNT paper
    """
    def __init__(self):
        super().__init__()

    def step(self, encoder_step, predictor_step):
        """
        Args:
            encoder_step: (B, V+1)
            predictor_step: (B, V+1)

        Returns:
            logits: (B, V+1)
        """
        return encoder_step + predictor_step

    def forward(self, encoder_output, predictor_output):
        """
        Args:
            encoder_output: (B, T, V+1)
            predictor_output: (B, S+1, V+1)

        Returns:
            logits: (B, T, S+1, V+1), compatible with torchaudio RNNT loss
        """
        max_input_size = encoder_output.shape[1]
        max_target_size = predictor_output.shape[1]
        encoder_unsqueezed = encoder_output.unsqueeze(2).expand(-1, -1, max_target_size, -1) # (B, T, S+1, V+1)
        predictor_unsqueezed = predictor_output.unsqueeze(1).expand(-1, max_input_size, -1, -1) # (B, T, S+1, V+1)
        return encoder_unsqueezed + predictor_unsqueezed
