import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from turkish_names_mlp import load_data

OUTPUT_DIR = os.path.join(os.getcwd(), "hafta-4/output")


def create_bad_model(vocab_size, embedding_dim=2, hidden_dim=100):
    C = torch.randn(vocab_size, embedding_dim)
    W1 = torch.randn(3 * embedding_dim, hidden_dim)
    b1 = torch.zeros(hidden_dim)
    W2 = torch.randn(hidden_dim, vocab_size)
    b2 = torch.zeros(vocab_size)
    return C, W1, b1, W2, b2


def create_kaiming_model(vocab_size, embedding_dim=2, hidden_dim=100):
    C = torch.randn(vocab_size, embedding_dim)
    fan_in = 3 * embedding_dim

    W1 = torch.randn(fan_in, hidden_dim) * ((5 / 3) / (fan_in ** 0.5))
    b1 = torch.zeros(hidden_dim)

    W2 = torch.randn(hidden_dim, vocab_size) * 0.01
    b2 = torch.zeros(vocab_size)

    return C, W1, b1, W2, b2


def forward_once(X, Y, model):
    C, W1, b1, W2, b2 = model
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    hpreact = emb_flat @ W1 + b1
    h = torch.tanh(hpreact)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)


    return loss, hpreact, h


def save_histogram(values, title, filename, xlabel):
    plt.figure(figsize=(8, 5))
    plt.hist(values.detach().view(-1).numpy(), bins=100)
    plt.title(title)
    plt.xlabel(xlabel)

    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    Xtr, Ytr, *_ = load_data()
    vocab_size = int(Xtr.max().item()) + 1

    bad = create_bad_model(vocab_size)
    good = create_kaiming_model(vocab_size)

    with torch.no_grad():
        loss_bad, hp_bad, h_bad = forward_once(Xtr, Ytr, bad)
        loss_good, hp_good, h_good = forward_once(Xtr, Ytr, good)

    sat_bad = (h_bad.abs() > 0.95).float().mean()
    sat_good = (h_good.abs() > 0.95).float().mean()

    deriv_bad = (1 - h_bad.pow(2)).mean()
    deriv_good = (1 - h_good.pow(2)).mean()

    print(f"Bad init loss={loss_bad.item():.4f} "
          f"\nsaturated={sat_bad.item():.4f} "
          f"\nmean tanh derivative={deriv_bad.item():.6f}")

    print(f"\nKaiming init loss={loss_good.item():.4f} "
          f"\nsaturated={sat_good.item():.4f} "
          f"\nmean tanh derivative={deriv_good.item():.6f}")

    save_histogram(
        hp_bad,
        "Bad Initialization - hpreact",
        "task5_bad_initialization_hpreact.png",
        "hpreact",
    )
    save_histogram(
        h_bad,
        "Bad Initialization - tanh",
        "task5_bad_initialization_tanh.png",
        "h",
    )
    save_histogram(
        hp_good,
        "Kaiming Initialization - hpreact",
        "task5_kaiming_initialization_hpreact.png",
        "hpreact",
    )
    save_histogram(
        h_good,
        "Kaiming Initialization - tanh",
        "task5_kaiming_initialization_tanh.png",
        "h",
    )



if __name__ == "__main__":
    main()
