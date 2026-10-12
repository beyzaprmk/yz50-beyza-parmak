import csv
from pathlib import Path

from data_loader import load_data, TURKISH_PATH, SHAKESPEARE_PATH

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "outputs" / "week8_final_results.csv"


def inspect_dataset(name, path):
    path = Path(path)
    if not path.is_file():
        print(f"File not found: {path}")
        return

    text = path.read_text(encoding="utf-8")
    train_data, val_data, tokenizer = load_data(path)
    print(f"Path: {path}")
    print(f"File size: {path.stat().st_size:,} bytes")
    print(f"Character count: {len(text):,}")
    print(f"Unique characters: {len(set(text))}")
    print(f"Tokenizer vocabulary: {tokenizer.vocab_size}")


def show_results():
    if not RESULTS_PATH.is_file():
        print(f"No final resultst")
        return

    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        print("Results CSV is empty.")
        return

    columns = [
        "dataset", "vocab_size", "parameters", "best_step",
        "train_loss", "val_loss", "training_seconds", "device",
    ]
    print(" | ".join(columns))
    for row in rows:
        print(" | ".join(row.get(column, "") for column in columns))

    valid = []
    for row in rows:
        try:
            valid.append((float(row["val_loss"]), row["dataset"]))
        except (ValueError, KeyError):
            pass

    if valid:
        print("\nLowest validation loss:", min(valid)[1], f"({min(valid)[0]:.4f})")

if __name__ == "__main__":
    inspect_dataset("Turkish", TURKISH_PATH)
    inspect_dataset("English", SHAKESPEARE_PATH)
    show_results()
