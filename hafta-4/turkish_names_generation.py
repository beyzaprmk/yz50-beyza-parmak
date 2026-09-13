import torch
import torch.nn as nn
import torch.nn.functional as F

from turkish_names_mlp import load_data, create_model, train, evaluate

# TASK 7 
def sample_mlp(model, stoi, itos, num_samples=20):
    C, W1, b1, W2, b2 = model[:5]
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            context = [stoi["."]] * 3
            name = ""

            while True:
                x = torch.tensor([context])
                emb = C[x]
                emb_flat = emb.view(1, -1)

                h = torch.tanh(emb_flat @ W1 + b1)
                logits = h @ W2 + b2
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, 1).item()
                ch = itos[ix]

                if ch == ".":
                    break

                name += ch
                context = context[1:] + [ix]

            samples.append(name)

    return samples


def build_bigram_data(words, stoi):
    X = []
    Y = []

    for word in words:
        context = stoi["."]

        for ch in word + ".":
            target = stoi[ch]
            X.append(context)
            Y.append(target)
            context = target

    return torch.tensor(X), torch.tensor(Y)


class BigramNN(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embedding(x)


def train_bigram(model, Xtr, Ytr, steps=10000, learning_rate=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for _ in range(steps):
        logits = model(Xtr)
        loss = F.cross_entropy(logits, Ytr)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate_bigram(model, X, Y):
    with torch.no_grad():
        logits = model(X)
        return F.cross_entropy(logits, Y).item()


def load_words():
    words = []

    with open("turkish_names.txt", "r", encoding="utf-8") as f:
        for word in f:
            word = word.strip()
            word = word.replace("I", "ı").replace("İ", "i").lower()
            words.append(word)

    return words


def sample_bigram(model, stoi, itos, num_samples=20):
    samples = []

    with torch.no_grad():
        for _ in range(num_samples):
            ix = stoi["."]
            name = ""

            while True:
                x = torch.tensor([ix])
                logits = model(x)
                probs = F.softmax(logits, dim=1)

                ix = torch.multinomial(probs, 1).item()
                ch = itos[ix]

                if ch == ".":
                    break

                name += ch

            samples.append(name)

    return samples


def main():
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = load_data()
    vocab_size = len(stoi)
    print("\nMLP eğitiliyor")
    mlp_model = create_model(vocab_size)

    train(
        mlp_model,
        Xtr,
        Ytr,
        steps=10000,
        learning_rate=0.2,
    )

    mlp_dev_loss = evaluate(Xdev, Ydev, mlp_model)
    mlp_samples = sample_mlp(
        mlp_model,
        stoi,
        itos,
        num_samples=20,
    )

    print(f"MLP dev loss: {mlp_dev_loss:.4f}")
    print("\nBigram modeli eğitiliyor")
    words = load_words()

    X_bigram, Y_bigram = build_bigram_data(words, stoi)

    g = torch.Generator().manual_seed(42)
    permutation = torch.randperm(
        X_bigram.shape[0],
        generator=g,
    )

    X_bigram = X_bigram[permutation]
    Y_bigram = Y_bigram[permutation]

    n = X_bigram.shape[0]
    n_train = int(0.80 * n)
    n_dev = int(0.10 * n)

    Xbtr = X_bigram[:n_train]
    Ybtr = Y_bigram[:n_train]
    Xbdev = X_bigram[n_train:n_train + n_dev]
    Ybdev = Y_bigram[n_train:n_train + n_dev]

    bigram_model = BigramNN(vocab_size)

    train_bigram(
        bigram_model,
        Xbtr,
        Ybtr,
        steps=10000,
        learning_rate=0.1,
    )

    bigram_dev_loss = evaluate_bigram(
        bigram_model,
        Xbdev,
        Ybdev,
    )

    bigram_samples = sample_bigram(
        bigram_model,
        stoi,
        itos,
        num_samples=20,
    )

    print(f"Bigram dev loss: {bigram_dev_loss:.4f}")

    print(f"\n{'MLP':<28} Bigram")

    for mlp_name, bigram_name in zip(mlp_samples, bigram_samples):
        print(f"{mlp_name:<28} {bigram_name}")

    print("\nDev loss")
    print(f"MLP     : {mlp_dev_loss:.4f}")
    print(f"Bigram  : {bigram_dev_loss:.4f}")


if __name__ == "__main__":
    main()
