import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        self.lookup = nn.Embedding(vocab_size,vocab_size)

    def forward(self, idx, targets=None):

        logits = self.lookup(idx)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view( B * T, C)

            targets = targets.view( B * T)

            loss = F.cross_entropy(logits,targets
            )

        return logits, loss