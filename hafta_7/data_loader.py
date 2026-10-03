import torch

from tokenizer import CharacterTokenizer


DATA_PATH = "data/Tiny_Shakespeare_input.txt"

TRAIN_RATIO = 0.90


def load_data(path=DATA_PATH):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # Character tokenizer
    tokenizer = CharacterTokenizer(text)

    data = torch.tensor(
        tokenizer.encode(text),
        dtype=torch.long
    )

    # %90 train %10 validation
    n = int(TRAIN_RATIO * len(data))

    train_data = data[:n]
    val_data = data[n:]

    return train_data, val_data, tokenizer


def get_batch(split,batch_size,block_size,train_data, val_data):
  
    data = train_data if split == "train" else val_data

    # random initlizaion
    ix = torch.randint(
        0,
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([data[i:i + block_size]for i in ix ])

    y = torch.stack([
        data[i + 1:i + block_size + 1]
        for i in ix
    ])

    return x, y

