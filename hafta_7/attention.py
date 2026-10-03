import torch
import torch.nn as nn
import torch.nn.functional as F

# causal=False -> Gelecekteki tokenlar da görülebilir
# causal= True tokenleri goremez
class Head(nn.Module):

    def __init__(
        self,
        n_embd,
        head_size,
        block_size,
        dropout=0.0,
        causal=True
    ):
        super().__init__()

        self.key = nn.Linear(n_embd,head_size,bias=False
        )

        self.query = nn.Linear( n_embd,head_size,bias=False
        )

        self.value = nn.Linear(
            n_embd,
            head_size,
            bias=False
        )

        self.register_buffer(
            "tril",
            torch.tril(
                torch.ones(block_size, block_size)
            )
        )

        self.dropout = nn.Dropout(dropout)

        self.causal = causal

        # Sadece analiz amacıyla son attention
        # ağırlıklarını tutar.
        self.last_weights = None

    def forward(self, x):

        B, T, C = x.shape

        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        wei = q @ k.transpose(-2, -1)

        wei = wei * k.shape[-1] ** -0.5

        if self.causal:

            wei = wei.masked_fill(
                self.tril[:T, :T] == 0,
                float("-inf")
            )

        wei = F.softmax(
            wei,
            dim=-1
        )

        # Analiz için son attention weights
        self.last_weights = wei.detach()

        wei = self.dropout(wei)

        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):

    def __init__(self,n_embd,num_heads,block_size, dropout=0.0,causal=True
    ):
        super().__init__()

        assert n_embd % num_heads == 0

        head_size = n_embd // num_heads

        self.heads = nn.ModuleList([
            Head(n_embd=n_embd,head_size=head_size,block_size=block_size,dropout=dropout,causal=causal
            )
            for _ in range(num_heads)
        ])

        self.proj = nn.Linear(head_size * num_heads,n_embd
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        head_outputs = [head(x) for head in self.heads ]

        out = torch.cat(head_outputs, dim=-1
        )

        out = self.proj(out)

        out = self.dropout(out)

        return out

    def get_attention_weights(self):

        return [head.last_weights for head in self.heads
        ]