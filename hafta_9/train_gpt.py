import csv
import sys
import time
from pathlib import Path

import torch

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WEEK8_DIR = PROJECT_ROOT / "hafta_8"
sys.path.insert(0, str(WEEK8_DIR))
from model import GPTLanguageModel, count_parameters
sys.path.remove(str(WEEK8_DIR))
from data_loader import load_data, get_batch, TURKISH_PATH

SEED = 42
BATCH_SIZE = 64
BLOCK_SIZE = 128
N_EMBD = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1
LEARNING_RATE = 3e-4
MAX_STEPS = 1000
EVAL_INTERVAL = 250
EVAL_BATCHES = 5
GENERATE_TOKENS = 500
VOCAB_SIZE = 280

OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "week9_bpe_results.csv"
SAMPLES_PATH = OUTPUT_DIR / "week9_generated_sample.txt"


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_fixed_eval_batches(data, batch_size, block_size, n_batches, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(0, len(data) - block_size, (n_batches, batch_size), generator=generator)
    batches = []
    for row in starts:
        x = torch.stack([data[int(i):int(i) + block_size] for i in row])
        y = torch.stack([data[int(i) + 1:int(i) + block_size + 1] for i in row])
        batches.append((x, y))
    return batches


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    losses = []
    for x, y in batches:
        _, loss = model(x.to(device), y.to(device))
        losses.append(loss.item())
    return sum(losses) / len(losses)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    device = select_device()

    train_data, val_data, tokenizer, raw_text = load_data(
        TURKISH_PATH, vocab_size=VOCAB_SIZE
    )
    train_eval = make_fixed_eval_batches(
        train_data, BATCH_SIZE, BLOCK_SIZE, EVAL_BATCHES, SEED + 1
    )
    val_eval = make_fixed_eval_batches(
        val_data, BATCH_SIZE, BLOCK_SIZE, EVAL_BATCHES, SEED + 2
    )

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
        dropout=DROPOUT,
        causal=True,
        use_feedforward=True,
        use_residual=True,
        use_layernorm=True,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    print(f"Characters: {len(raw_text):,} | UTF-8 bytes: {len(raw_text.encode('utf-8')):,}")
    print(f"Tokenizer vocab: {tokenizer.vocab_size} | Train tokens: {len(train_data):,} | Val tokens: {len(val_data):,}")
    print(f"Compression (bytes/token): {tokenizer.compression_stats(raw_text)['bytes_per_token']:.3f}")
    print(f"Parameters: {count_parameters(model):,} | Device: {device}")

    best_val = float("inf")
    best_state = None
    best_step = 0
    best_train = float("inf")
    start = time.perf_counter()

    for step in range(1, MAX_STEPS + 1):
        model.train()
        x, y = get_batch("train", BATCH_SIZE, BLOCK_SIZE, train_data, val_data)
        _, loss = model(x.to(device), y.to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == MAX_STEPS:
            train_loss = evaluate(model, train_eval, device)
            val_loss = evaluate(model, val_eval, device)
            if val_loss < best_val:
                best_val, best_train, best_step = val_loss, train_loss, step
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"step {step:5d} | train {train_loss:.4f} | val {val_loss:.4f} | best val {best_val:.4f} (step {best_step})")


  
    model.load_state_dict(best_state)
    model.eval()
    checkpoint_path = CHECKPOINT_DIR / "week9_turkish_bpe_best.pt"
    torch.save({
        "model_state_dict": best_state,
        "vocab_size": tokenizer.vocab_size,
        "block_size": BLOCK_SIZE,
        "config": {
            "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
            "dropout": DROPOUT, "vocab_size": tokenizer.vocab_size,
        },
        "tokenizer_path": str(OUTPUT_DIR / "turkish_bpe.json"),
    }, checkpoint_path)

    # Start with the first UTF-8 byte of the source
    start_id = tokenizer.encode(raw_text[:1])[0]
    context = torch.tensor([[start_id]], dtype=torch.long, device=device)
    generated_ids = model.generate(context, GENERATE_TOKENS)[0].cpu().tolist()
    sample = tokenizer.decode(generated_ids)
    SAMPLES_PATH.write_text(sample, encoding="utf-8")

    result = {
        "dataset": "Turkish",
        "characters": len(raw_text),
        "utf8_bytes": len(raw_text.encode("utf-8")),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_data),
        "validation_tokens": len(val_data),
        "bytes_per_token": round(tokenizer.compression_stats(raw_text)["bytes_per_token"], 6),
        "n_embd": N_EMBD, "n_head": N_HEAD, "n_layer": N_LAYER,
        "block_size": BLOCK_SIZE, "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE, "max_steps": MAX_STEPS,
        "parameters": count_parameters(model), "best_step": best_step,
        "train_loss": round(best_train, 6), "val_loss": round(best_val, 6),
        "training_seconds": round(time.perf_counter() - start, 2),
        "device": str(device), "checkpoint": str(checkpoint_path),
    }
    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(result.keys()))
        writer.writeheader()
        writer.writerow(result)

    print(f"\ngenerated sample:\n{sample[:1000]}")
    
if __name__ == "__main__":
    main()
