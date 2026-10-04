import torch

from data_loader import load_data
from bigram import BigramLanguageModel
from model import GPTLanguageModel


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BLOCK_SIZE = 64
N_EMBD = 64

NUM_TOKENS = 500
START_TEXT = "\n"


train_data, val_data, tokenizer = load_data()

vocab_size = tokenizer.vocab_size

def generate_gpt(model, start_text, num_tokens):
    tokens = tokenizer.encode(start_text)

    idx = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    generated = model.generate(
        idx,
        max_new_tokens=num_tokens
    )

    return tokenizer.decode(generated[0].tolist())


def generate_bigram(model, start_text, num_tokens):
    tokens = tokenizer.encode(start_text)

    idx = torch.tensor(
        [tokens],
        dtype=torch.long,
        device=DEVICE
    )

    for _ in range(num_tokens):

        logits, _ = model(idx)

        logits = logits[:, -1, :]

        probs = torch.softmax(logits, dim=-1)

        next_token = torch.multinomial(
            probs,
            num_samples=1
        )

        idx = torch.cat(
            (idx, next_token),
            dim=1
        )

    return tokenizer.decode(idx[0].tolist())
results = []


bigram = BigramLanguageModel(vocab_size).to(DEVICE)

bigram.load_state_dict(
    torch.load(
        "outputs/bigram.pt",
        map_location=DEVICE
    )
)

bigram.eval()

text = generate_bigram(
    bigram,
    START_TEXT,
    NUM_TOKENS
)

results.append(
    f"BIGRAM\n"
    
    f"{text}\n"
)


single_head = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=1
).to(DEVICE)

single_head.load_state_dict(
    torch.load(
        "outputs/model_1head.pt",
        map_location=DEVICE
    )
)

single_head.eval()

text = generate_gpt(
    single_head,
    START_TEXT,
    NUM_TOKENS
)

results.append(
    f"SINGLE HEAD\n"
    f"{text}\n"
)


multi_head = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=4
).to(DEVICE)

multi_head.load_state_dict(
    torch.load(
        "outputs/model_4head.pt",
        map_location=DEVICE
    )
)

multi_head.eval()

text = generate_gpt(
    multi_head,
    START_TEXT,
    NUM_TOKENS
)

results.append(
    f"MULTI HEAD\n"
    f"{text}\n"
)


# Multi Head - No Mask
noncausal = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=BLOCK_SIZE,
    n_embd=N_EMBD,
    n_head=4,
    causal=False
).to(DEVICE)

noncausal.load_state_dict(
    torch.load(
        "outputs/model_4head_noncausal.pt",
        map_location=DEVICE
    )
)

noncausal.eval()

text = generate_gpt(
    noncausal,
    START_TEXT,
    NUM_TOKENS
)

results.append(
    f"MULTI HEAD - NO MASK\n"
    f"{text}\n"
)


output_path = "outputs/generated_text.txt"

with open(output_path, "w", encoding="utf-8") as f:
    f.write("\n".join(results))

print(f"\nGenerated text saved to: {output_path}")