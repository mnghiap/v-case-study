import torch
import torch.nn as nn

class BLSTM_Entity_Recognizer(nn.Module):
    """
    Bi-directional LM to "recognize" entities in an transcription
    """
    def __init__(self, n_blstm_layers, input_dim, embed_dim, hidden_dim, output_dim):
        super().__init__()
        self.n_blstm_layers = n_blstm_layers
        self.embed = nn.Embedding(input_dim, embed_dim)
        self.blstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            n_blstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.final_linear = nn.Linear(2*hidden_dim, output_dim)


    def forward(self, x):
        """
        Forward whole sequence (withour any BOS, EOS)
        to predict whether a token is entity or not

        Args:
            x: target sequence WITHOUT BOS (B, S)
        
        Returns:
            logits: unnormalized logits (B, S, output dim)
        """
        x = self.embed(x)
        x, _ = self.blstm(x)
        logits = self.final_linear(x)
        return logits
    

    def compute_ce_loss(self, x, target):
        """
        Compute the cross entropy loss between x and target

        Args:
            x: (B, S)

        Returns:
            ce: CE loss (B, S)
        """
        logits = self(x).log_softmax(-1) # (B, S, out dim)
        ce = nn.functional.cross_entropy(
            input=logits.transpose(1, 2),
            target=target,
            reduction="none",
        )
        return ce
