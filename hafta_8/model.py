import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Head, MultiHeadAttention


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        block_size,
        dropout=0.0,
        causal=True,
        use_feedforward=True,
        use_residual=True,
        use_layernorm=True,
    ):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head.")

        head_size = n_embd // n_head
        self.attention = (   Head(n_embd, head_size, block_size, dropout, causal)
            if n_head == 1
            else MultiHeadAttention(n_embd, n_head, block_size, dropout, causal)
        )
        self.ffwd = FeedForward(n_embd, dropout) if use_feedforward else None
        self.use_residual = use_residual
        self.use_layernorm = use_layernorm
        self.ln1 = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.ln2 = (nn.LayerNorm(n_embd)
            if use_layernorm and use_feedforward
            else nn.Identity()
        )

    def forward(self, x):
        attn_input = self.ln1(x) if self.use_layernorm else x
        attn_out = self.attention(attn_input)
        x = x + attn_out if self.use_residual else attn_out

        if self.ffwd is not None:
            ff_input = self.ln2(x) if self.use_layernorm else x
            ff_out = self.ffwd(ff_input)
            x = x + ff_out if self.use_residual else ff_out

        return x

    def get_attention_weights(self):
        if hasattr(self.attention, "get_attention_weights"):
            return self.attention.get_attention_weights()
        return [self.attention.last_weights]


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        block_size,
        n_embd=64,
        n_head=4,
        n_layer=1,
        dropout=0.1,
        causal=True,
        use_feedforward=True,
        use_residual=True,
        use_layernorm=True,
    ):
        super().__init__()
        if n_layer < 1:
            raise ValueError("n_layer must be at least 1.")

        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        
        self.blocks = nn.Sequential(*[
            TransformerBlock(
                n_embd=n_embd,
                n_head=n_head,
                block_size=block_size,
                dropout=dropout,
                causal=causal,
                use_feedforward=use_feedforward,
                use_residual=use_residual,
                use_layernorm=use_layernorm,
            )

            for _ in range(n_layer)
        ])
        self.final_ln = nn.LayerNorm(n_embd) if use_layernorm else nn.Identity()
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)


    def forward(self, idx, targets=None):
        _, time_steps = idx.shape
        if time_steps > self.block_size:
            raise ValueError(
                f"Sequence length {time_steps} exceeds block_size={self.block_size}."
            )

        positions = torch.arange(time_steps, device=idx.device)
        x = self.token_embedding(idx) + self.position_embedding(positions)
        x = self.blocks(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        was_training = self.training
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            probs = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        self.train(was_training)
        return idx

    def get_attention_weights(self):
        return [block.get_attention_weights()
            for block in self.blocks
            if hasattr(block, "get_attention_weights")
        ]


def count_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters())
