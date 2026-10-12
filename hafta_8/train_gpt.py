
import csv
import time
from pathlib import Path

import torch

from data_loader import load_data, get_batch, TURKISH_PATH, SHAKESPEARE_PATH
from model import GPTLanguageModel, count_parameters


seed = 42
batch_size = 64
block_size = 128
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 3e-4
max_step = 1000
eval_interval = 250
eval_batche = 5
generate_tokens = 500

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
RESULTS_PATH = OUTPUT_DIR / "week8_final_results.csv"
SAMPLES_PATH = OUTPUT_DIR / "week8_generated_samples.txt"


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def make_fixed_eval_batches(data, batch_size, block_size, n_batches, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    starts = torch.randint(
        0, len(data) - block_size, (n_batches, batch_size), generator=generator
    )
    batches = []
    for row in starts:
        x = torch.stack([data[int(i):int(i) + block_size] for i in row])
        y = torch.stack([
            data[int(i) + 1:int(i) + block_size + 1] for i in row
        ])
        batches.append((x, y))
    return batches


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    values = []
    for x, y in batches:
        _, loss = model(x.to(device), y.to(device))
        values.append(loss.item())
    return sum(values) / len(values)


def train_corpus(name, path, device):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{name} dataset not found: {path}")

    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    raw_text = path.read_text(encoding="utf-8")
    train_data, val_data, tokenizer = load_data(path)

    train_eval_batches = make_fixed_eval_batches( train_data, batch_size, block_size, eval_batche, seed + 1 )
    val_eval_batches = make_fixed_eval_batches(   val_data, batch_size, block_size, eval_batche, seed + 2)

    model = GPTLanguageModel(vocab_size=tokenizer.vocab_size,  block_size=block_size,
        n_embd=n_embd,  n_head=n_head,  n_layer=n_layer,
        dropout=dropout,  causal=True,   use_feedforward=True,
        use_residual=True,  use_layernorm=True,  ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    parameters = count_parameters(model)

    print(f"Dataset: {name}")
    print(f"Characters: {len(raw_text):,}")
    print(f"Unique characters: {len(set(raw_text))} / {tokenizer.vocab_size}")
    print(f"Train tokens: {len(train_data):,} | Validation tokens: {len(val_data):,}")
    print(
        f"Config: n_embd={n_embd}, n_head={n_head}, n_layer={n_layer}, "
        f"block_size={block_size}, batch_size={batch_size}, dropout={dropout}"
    )
    print(f"Parameters: {parameters:,} | Device: {device}")

    best_val = float("inf")
    best_train = float("inf")
    best_step = 0
    best_state = None

    synchronize(device)
    start = time.perf_counter()

    for step in range(1, max_step + 1):
        model.train()
        x, y = get_batch("train", batch_size, block_size, train_data, val_data)
        x, y = x.to(device), y.to(device)

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step % eval_interval == 0 or step == max_step:
            train_loss = evaluate(model, train_eval_batches, device)
            val_loss = evaluate(model, val_eval_batches, device)

            if val_loss < best_val:
                best_val = val_loss
                best_train = train_loss
                best_step = step
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }

            print(
                f"step {step:5d} | train {train_loss:.4f} | "
                f"val {val_loss:.4f} | best val {best_val:.4f} (step {best_step})"
            )

    synchronize(device)
    elapsed = time.perf_counter() - start
    if best_state is None:
        raise RuntimeError(f"No checkpoint was selected for {name}.")

    model.load_state_dict(best_state)
    model.eval()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = name.lower().replace(" ", "_")
    checkpoint_path = CHECKPOINT_DIR / f"week8_{safe_name}_best.pt"
    torch.save(best_state, checkpoint_path)

    start_text = "\n" if "\n" in tokenizer.stoi else raw_text[0]
    context = torch.tensor(
        [[tokenizer.stoi[start_text]]], dtype=torch.long, device=device
    )
    generated = model.generate(context, generate_tokens)[0].detach().cpu().tolist()
    sample = tokenizer.decode(generated)

    result = {
        "dataset": name,
        "data_path": str(path),
        "file_bytes": path.stat().st_size,
        "characters": len(raw_text),
        "unique_characters": len(set(raw_text)),
        "vocab_size": tokenizer.vocab_size,
        "train_tokens": len(train_data),
        "validation_tokens": len(val_data),
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
        "batch_size": batch_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "max_steps": max_step,
        "parameters": parameters,
        "best_step": best_step,
        "train_loss": round(best_train, 6),
        "val_loss": round(best_val, 6),
        "training_seconds": round(elapsed, 2),
        "device": str(device),
        "checkpoint": str(checkpoint_path),
    }

    print(f"\n{name} generated sample:\n{sample[:600]}\n")
    return result, sample


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = select_device()
    datasets = [
        ("Shakespeare", SHAKESPEARE_PATH),
        ("Turkish", TURKISH_PATH),
    ]

    results = []
    samples = []

    for name, path in datasets:
        result, sample = train_corpus(name, path, device)
        results.append(result)
        samples.append(
            f"BEST VAL LOSS: {result['val_loss']:.6f}\n"
            f"BEST STEP: {result['best_step']}\n{sample}\n"
        )

    with RESULTS_PATH.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    SAMPLES_PATH.write_text("\n\n".join(samples), encoding="utf-8")

  
    print(
        f"{'Dataset':<14} {'Vocab':>7} {'Params':>10} {'Best step':>10} "
        f"{'Train loss':>12} {'Val loss':>10} {'Seconds':>10}"
    )
    for row in results:
        print(
            f"{row['dataset']:<14} {row['vocab_size']:>7} "
            f"{row['parameters']:>10,} {row['best_step']:>10} "
            f"{row['train_loss']:>12.4f} {row['val_loss']:>10.4f} "
            f"{row['training_seconds']:>10.2f}"
        )

    print(f"\nsaved result: {RESULTS_PATH}")
    print(f"Saved samples: {SAMPLES_PATH}")


if __name__ == "__main__":
    main()
