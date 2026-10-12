import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, n_embd, head_size, block_size, dropout=0.0, causal=True):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.attention_dropout = nn.Dropout(dropout)
        self.causal = causal
        self.last_weights = None

    def forward(self, x):
        _, time_steps, _ = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        weights = q @ k.transpose(-2, -1)
        weights = weights * (k.shape[-1] ** -0.5)

        if self.causal:
            weights = weights.masked_fill(
                self.tril[:time_steps, :time_steps] == 0,
                float("-inf"),
            )

        weights = F.softmax(weights, dim=-1)
        self.last_weights = weights.detach()
        weights = self.attention_dropout(weights)
        return weights @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, block_size, dropout=0.0, causal=True):
        super().__init__()
        if n_embd % num_heads != 0:
            raise ValueError(" must be divisible ")

        head_size = n_embd // num_heads
        self.heads = nn.ModuleList([
            Head( n_embd=n_embd, head_size=head_size, block_size=block_size, dropout=dropout,  causal=causal, )
            for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_embd, n_embd)
        self.projection_dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.projection_dropout(self.proj(out))


    def get_attention_weights(self):
        return [head.last_weights for head in self.heads]
