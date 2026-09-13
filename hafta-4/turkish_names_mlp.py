import torch
import torch.nn.functional as F

#baglam uzunlugu
BLOCK_SIZE = 3

EMBEDDING_DIM = 2
HIDDEN_DIM = 100
BATCH_SIZE = 64
DATA_PATH = "turkish_names.txt"


def load_data():
    words = []
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for w in f:
            w = w.strip()
            w = w.replace("I", "ı").replace("İ", "i").lower()
            words.append(w)

    chars = ["."] + sorted(set("".join(words)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}

    X, Y = [], []

    for word in words:
        context = [0] * BLOCK_SIZE
        
        for ch in word + ".":
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            #sliding window
            context = context[1:] + [ix]

    X = torch.tensor(X)
    Y = torch.tensor(Y)
    #shuffle dataset
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(X.shape[0], generator=g)

    X, Y = X[perm], Y[perm]

    n = X.shape[0]
    n_train = int(0.80 * n)
    n_dev = int(0.10 * n)

    return (
        X[:n_train], Y[:n_train],
        X[n_train:n_train+n_dev], Y[n_train:n_train+n_dev],
        X[n_train+n_dev:], Y[n_train+n_dev:],
        stoi, itos
    )


def create_model(vocab_size, embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM):
    C = torch.randn(vocab_size, embedding_dim)
    W1 = torch.randn(BLOCK_SIZE * embedding_dim, hidden_dim) * 0.1
    b1 = torch.randn(hidden_dim) * 0.1
    W2 = torch.randn(hidden_dim, vocab_size) * 0.1
    b2 = torch.randn(vocab_size) * 0.1

    parameters = [C, W1, b1, W2, b2]
    for p in parameters:
        p.requires_grad = True

    return C, W1, b1, W2, b2, parameters


def forward(X, Y, C, W1, b1, W2, b2):
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    #scores
    logits = h @ W2 + b2
    #probability converted scores
    loss = F.cross_entropy(logits, Y)
    return loss, logits


def evaluate(X, Y, model):
    with torch.no_grad():
        return forward(X, Y, *model[:5])[0].item()


def train(model, Xtr, Ytr, steps, learning_rate, batch_size=BATCH_SIZE):
    parameters = model[5]

    for step in range(steps):
        ix = torch.randint(0, Xtr.shape[0], (batch_size,))
        loss, _ = forward(Xtr[ix], Ytr[ix], *model[:5])

        for p in parameters:
            p.grad = None

        loss.backward()

        with torch.no_grad():
            for p in parameters:
                p -= learning_rate * p.grad

    return model


def task1_2():
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = load_data()
    vocab_size = len(stoi)

    print("Train:", Xtr.shape, Ytr.shape)
    print("Dev:  ", Xdev.shape, Ydev.shape)
    print("Test: ", Xte.shape, Yte.shape)
    print("Vocabulary size:", vocab_size)
    print("Characters:", list(stoi.keys()))

    model = create_model(vocab_size)

    C, W1, b1, W2, b2, _ = model
    emb = C[Xtr]
    emb_flat = emb.view(emb.shape[0], -1)
    hpreact = emb_flat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2

    #NLL
    probs = logits.exp() / logits.exp().sum(dim=1, keepdim=True)
    correct_probs = probs[torch.arange(Xtr.shape[0]), Ytr]
    manual_loss = -correct_probs.log().mean()

    ce_loss = F.cross_entropy(logits, Ytr)

    print("\nShapes:")
    print("C:", C.shape)
    print("Embedding:", emb.shape)
    print("Flattened embedding:", emb_flat.shape)
    print("Hidden:", h.shape)
    print("Logits:", logits.shape)

    print("\nManual loss:", manual_loss.item())
    print("Cross entropy:", ce_loss.item())
    print("Difference:", abs(manual_loss.item() - ce_loss.item()))

    return model, (Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos)



def task3():
    model, data = task1_2()
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = data



    #tek bir minibatch'i overfit ederek backward zincirini doğrulama
    ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
    Xbatch, Ybatch = Xtr[ix], Ytr[ix]

    for step in range(1000):
        loss, _ = forward(Xbatch, Ybatch, *model[:5])

        for p in model[5]:
            p.grad = None

        loss.backward()

        with torch.no_grad():
            for p in model[5]:
                p -= 0.1 * p.grad

    print("Single-minibatch loss:", loss.item())

    #bütün train setini rastgele minibatch'lerle eğit.
    train(model, Xtr, Ytr, steps=10000, learning_rate=0.1)

    print("Train loss:", evaluate(Xtr, Ytr, model))
    print("Dev loss:  ", evaluate(Xdev, Ydev, model))
    print("Test loss: ", evaluate(Xte, Yte, model))


if __name__ == "__main__":
    task3()
