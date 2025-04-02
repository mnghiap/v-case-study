import torch
import torch.nn as nn

class Encoder(nn.Module):
    """
    Simple unidirectional LSTM encoder

    Important constraint: blank must always be the last in the output dim!

    Args:
        input_dim: Num audio features, e.g. n MFCC coefficients (F)
        hidden_dim: LSTM hidden size
        output_dim: Vocab + blank (V+1)
        n_lstm_layers: num LSTM layers
    """
    def __init__(self, input_dim, hidden_dim, output_dim, n_lstm_layers):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
        )
        self.final_linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x: (B, T, F)

        Returns:
            logits: unnormalized logits (B, T, V+1)
        """
        x, _ = self.lstm(x)
        logits = self.final_linear(x)
        return logits
