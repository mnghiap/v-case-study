import torch
import torch.nn as nn

class LSTM_LM(nn.Module):
    """
    Simple LSTM LM, almost identical to predictor network of RNNT

    Args:
        bos_idx:
        n_lstm_layers
        input_dim: Vocab size (V)
        embed_dim:
        hidden_dim:
        output_dim: V
    """
    def __init__(self, bos_idx, eos_idx, n_lstm_layers, input_dim, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
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
    

    def get_bos_symbol(self, batch_size):
        return torch.full((batch_size, 1), fill_value=self.bos_idx)


    def step(self, x, state):
        """
        Forward but just one single step, with state as input.
        Mainly used for decoding.
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
        batch_size = x.shape[0]
        x = torch.cat([torch.full((batch_size, 1), self.bos_idx), x], dim=1)
        x = self.embed(x)
        predictor_state = self.get_initial_state(batch_size)
        x, (hn, cn) = self.lstm(x, predictor_state)
        logits = self.final_linear(x)
        return logits, (hn, cn)

    
    def compute_lm_score(self, x: torch.Tensor):
        """
        Args:
            x: target sequences without BOS and EOS (B, S)
        Returns:
            scores: log p(seq) (B,)
        """
        x_eos = torch.cat([x, torch.full((x.shape[0], 1), self.eos_idx)], dim=1)
        logits, _ = self.forward(x) # (B, S, V)
        ce = nn.functional.cross_entropy( # -log p(seq)
            input=logits.transpose(1, 2), # (B, V, S)
            target=x_eos, # (B, S)
            reduction="none",
        )
        scores = ce.sum(dim=1) # (B,)
        return -scores