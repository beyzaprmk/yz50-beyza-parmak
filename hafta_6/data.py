import torch

DATA_PATH = "./turkish_names.txt"
SEED = 42

def load_words(path=DATA_PATH):

    with open(path, "r", encoding="utf-8") as f:
        words = [
            line.strip()
            for line in f
            if line.strip()
        ]

    # Turkish character normalization
    words = [ w.replace("I", "ı")  .replace("İ", "i")  .lower() for w in words ]

    return words


def build_vocabulary(words):

    chars = ["."] + sorted(
        set("".join(words))
    )

    stoi = {
        ch: i
        for i, ch in enumerate(chars)
    }

    itos = { i: ch for ch, i in stoi.items() }

    return stoi, itos

def build_dataset(words, stoi, block_size):

    X = []
    Y = []

    for word in words:

        context = [0] * block_size

        for ch in word + ".":

            ix = stoi[ch]

            X.append(context)
            Y.append(ix)

            context = context[1:] + [ix]

    return ( torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long))


def split_dataset(X, Y, seed=SEED):

    n = X.shape[0]

    g = torch.Generator().manual_seed(seed)

    indices = torch.randperm( n, generator=g)

    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)

    train_idx = indices[:n_train]

    dev_idx = indices[  n_train:n_train + n_dev ]

    test_idx = indices[  n_train + n_dev:  ]

    return (
        X[train_idx],
        Y[train_idx],
        X[dev_idx],
        Y[dev_idx],
        X[test_idx],
        Y[test_idx]
    )

def load_data(block_size, path=DATA_PATH, seed=SEED
):

    words = load_words(path)

    stoi, itos = build_vocabulary(
        words
    )

    X, Y = build_dataset( words, stoi, block_size)

    (Xtr, Ytr, Xdev, Ydev, Xte, Yte) = split_dataset( X, Y, seed)

    return (   Xtr,  Ytr, Xdev,Ydev, Xte, Yte, len(stoi), stoi, itos)