from pathlib import Path
import torch
from tokenizer import BPETokenizer

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
WEEK8_DIR = PROJECT_ROOT / "hafta_8"
TURKISH_PATH = WEEK8_DIR / "data" / "kuyruksuz_tilki.txt"
TRAIN_RATIO = 0.90
TOKENIZER_PATH = BASE_DIR / "outputs" / "turkish_bpe.json"


def load_data(path=TURKISH_PATH, vocab_size=280, tokenizer_path=TOKENIZER_PATH):
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path

    text = path.read_text(encoding="utf-8")


    tokenizer_path = Path(tokenizer_path)
    tokenizer = BPETokenizer()
    split_char = int(TRAIN_RATIO * len(text))
    train_text = text[:split_char]
  
    training_sample = train_text.encode("utf-8")[:100_000].decode("utf-8", errors="ignore")
    tokenizer.train(training_sample, vocab_size=vocab_size)
    tokenizer.save(tokenizer_path)

    all_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    split_index = int(TRAIN_RATIO * len(all_ids))
    train_data, val_data = all_ids[:split_index], all_ids[split_index:]
   
    return train_data, val_data, tokenizer, text


def get_batch(split, batch_size, block_size, train_data, val_data):
    
    data = train_data if split == "train" else val_data
    if len(data) <= block_size:
        raise ValueError(
            f"{split} data ({len(data)} tokens) must exceed block_size ({block_size})."
        )
    starts = torch.randint(0, len(data) - block_size, (batch_size,))
    x = torch.stack([data[int(i):int(i) + block_size] for i in starts])
    y = torch.stack([data[int(i) + 1:int(i) + block_size + 1] for i in starts])
    return x, y
