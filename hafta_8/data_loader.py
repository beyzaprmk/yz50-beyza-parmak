from pathlib import Path
import torch
from tokenizer import CharacterTokenizer


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TURKISH_PATH = DATA_DIR / "kuyruksuz_tilki.txt"
SHAKESPEARE_PATH = (
    BASE_DIR.parent / "hafta_7" / "data" / "Tiny_Shakespeare_input.txt"
)
TRAIN_RATIO = 0.90


def load_data(path=TURKISH_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")

    text = path.read_text(encoding="utf-8")
    if len(text) < 2:
        raise ValueError(f"Data file is empty or too short: {path}")

    tokenizer = CharacterTokenizer(text)
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)

    split_index = int(TRAIN_RATIO * len(data))
    train_data = data[:split_index]
    val_data = data[split_index:]

    if len(train_data) < 2 or len(val_data) < 2:
        raise ValueError("The train/validation split is too small.")

    return train_data, val_data, tokenizer


def get_batch(split, batch_size, block_size, train_data, val_data):
    if split not in {"train", "val", "validation"}:
        raise ValueError("split must be 'train' or 'val'.")

    data = train_data if split == "train" else val_data
    if len(data) <= block_size:
        raise ValueError(
            f"{split} data ({len(data)} tokens) must exceed block_size ({block_size})."  )

    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[int(i):int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + block_size + 1] for i in starts ])
    return x, y
