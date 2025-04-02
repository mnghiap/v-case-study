import torch
import torch.nn as nn

class Predictor(nn.Module):
    """
    Simple unidirectional LSTM predictor network for RNNT.

    Important constraint: blank must always be the last in the output dim!

    Args:
        bos_idx:
        n_lstm_layers
        input_dim: Vocab size (V)
        embed_dim:
        hidden_dim:
        output_dim: Vocab + blank (V+1)
    """
    def __init__(self, bos_idx, n_lstm_layers, input_dim, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.bos_idx = bos_idx
        self.n_lstm_layers = n_lstm_layers
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_lstm_layers,
            batch_first=True,
        )
        self.final_linear = nn.Linear(hidden_dim, output_dim)

    
    def get_initial_state(self, batch_size):
        h0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.n_lstm_layers, batch_size, self.hidden_dim)
        return (h0, c0)


    def get_initial_predictor_symbol(self, batch_size):
        return torch.full((batch_size, 1), fill_value=self.bos_idx)


    def step(self, x, state):
        """
        Args:
            x: (B, S), S = step length, usually 1
            state: LSTM state (n_lstm_layers, B, hidden_dim)

        Returns:
            logits: (B, S, V+blank)
            (hn, cn): Tuple of states ((n_lstm_layers, B, hidden_dim), (n_lstm_layers, B, hidden_dim))
        """
        x = self.embed(x)
        x, (hn, cn) = self.lstm(x, state)
        logits = self.final_linear(x)
        return logits, (hn, cn)


    def forward(self, x):
        """
        Forward whole sequence

        Args:
            x: target sequence WITHOUT BOS (B, S)

        Returns:
            logits: unnormalized logits (B, S+1, V+1)
        """
        x = torch.cat([torch.full((x.shape[0], 1), self.bos_idx), x], dim=1)
        x = self.embed(x)
        predictor_state = self.get_initial_state(x.shape[0])
        x, (hn, cn) = self.lstm(x, predictor_state)
        logits = self.final_linear(x)
        return logits, (hn, cn)
