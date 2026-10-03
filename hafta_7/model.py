import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Head, MultiHeadAttention


class GPTLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd=64,
        n_head=1,
        dropout=0.0,
        causal=True
    ):
        super().__init__()

        self.block_size = block_size
        self.token_embedding = nn.Embedding(
            vocab_size,
            n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size,
            n_embd
        )

        if n_head == 1:

            self.attention = Head(
                n_embd=n_embd,
                head_size=n_embd,
                block_size=block_size,
                dropout=dropout,
                causal=causal
            )

        else:

            self.attention = MultiHeadAttention(
                n_embd=n_embd,
                num_heads=n_head,
                block_size=block_size,
                dropout=dropout,
                causal=causal
            )

        self.lm_head = nn.Linear(
            n_embd,
            vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        tok_emb = self.token_embedding(idx)

        pos = torch.arange(
            T,
            device=idx.device
        )

        pos_emb = self.position_embedding(pos)

        x = tok_emb + pos_emb

        x = self.attention(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits = logits.view(B * T,C)

            targets = targets.view(B * T )

            loss = F.cross_entropy(
                logits,
                targets
            )

        return logits, loss


    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens
    ):

        for _ in range(max_new_tokens):

            # Son block_size tokenı kullan.
            idx_cond = idx[:, -self.block_size:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]

            # Logits -> probabilities
            probs = F.softmax(
                logits,
                dim=-1
            )

            # Bir sonraki tokenı sample et.
            idx_next = torch.multinomial(
                probs,
                num_samples=1
            )

            # Sequence'e ekle.
            idx = torch.cat(
                (idx, idx_next),
                dim=1
            )

        return idx

    def get_attention_weights(self):

        if hasattr(
            self.attention,
            "get_attention_weights"
        ):
            return self.attention.get_attention_weights()

        return [
            self.attention.last_weights
        ]