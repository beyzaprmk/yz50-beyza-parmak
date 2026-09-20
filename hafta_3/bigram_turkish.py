import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F


data_path = "./turkish_names.txt"
output_path = "hafta-3/outputs/turkish_bigram_matrix.png"

number_of_sumbles = 20

epoches = 400
lr = 1.0 

smoothing = 1.0

seed = 2147483647


def turkish_lower(text: str) -> str:
  
    replacement_map = {
        "I": "ı",
        "İ": "i",
    }
    for tr_upper, tr_lower in replacement_map.items():
        text = text.replace(tr_upper, tr_lower)
    return text.lower()


def load_names(path):
    names = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            name = turkish_lower(line.strip())

          
            if not name:
                continue

            names.append(name)

    return names


def build_vocabulary(names):
    chars = sorted(set("".join(names)))

    # '.' = başlangıç bitiş tokenı
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi["."] = 0

    itos = {i: ch for ch, i in stoi.items()}

    return chars, stoi, itos


def count_bigrams_dictionary(names):
    bigram_counts = {}

    for name in names:
        chars = ["."] + list(name) + ["."]

        for ch1, ch2 in zip(chars, chars[1:]):
            bigram = (ch1, ch2)

            if bigram not in bigram_counts:
                bigram_counts[bigram] = 0

            bigram_counts[bigram] += 1

    return bigram_counts


def build_count_matrix(names, stoi):
    vocab_size = len(stoi)

    N = torch.zeros(
        (vocab_size, vocab_size),
        dtype=torch.int32
    )

    for name in names:
        chars = ["."] + list(name) + ["."]

        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            N[ix1, ix2] += 1

    return N


def visualize_matrix(
    N,
    itos,
    save_path=output_path
):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    plt.figure(figsize=(16, 16))
    plt.imshow(N, cmap="Blues")

    vocab_size = N.shape[0]

    for i in range(vocab_size):
        for j in range(vocab_size):
            ch1 = itos[i]
            ch2 = itos[j]

            if ch1 == " ":
                ch1 = "SPACE"
            if ch2 == " ":
                ch2 = "SPACE"

            plt.text(
                j,
                i,
                ch1 + ch2,
                ha="center",
                va="bottom",
                fontsize=6
            )

            plt.text(
                j,
                i,
                N[i, j].item(),
                ha="center",
                va="top",
                fontsize=5
            )

    plt.axis("off")
    plt.title("Turkish Character Bigram Frequency")
    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    print(f"Figure saved to: {save_path}")


def build_probability_matrix(N, smoothing=1.0):
    P = (N + smoothing).float()
    P /= P.sum(
        dim=1,
        keepdim=True
    )

    return P


def sample_count_model(
    P,
    itos,
    num_samples=20,
    seed=seed
):
    generator = torch.Generator().manual_seed(seed)
    results = []

    for _ in range(num_samples):
        result = []
        ix = 0

        while True:
            p = P[ix]

            ix = torch.multinomial(
                p,
                num_samples=1,
                replacement=True,
                generator=generator
            ).item()

            if ix == 0:
                break

            result.append(itos[ix])

        results.append("".join(result))

    return results


def calculate_count_nll(names, stoi, P):
    log_likelihood = 0.0
    n = 0

    for name in names:
        chars = ["."] + list(name) + ["."]

        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]

            prob = P[ix1, ix2]
            log_likelihood += torch.log(prob) #log(ab) = loga + logb 
            n += 1

    nll = -log_likelihood / n
    return nll


def build_training_data(names, stoi):
    xs = []
    ys = []

    for name in names:
        chars = ["."] + list(name) + ["."]

        for ch1, ch2 in zip(chars, chars[1:]):
            xs.append(stoi[ch1])
            ys.append(stoi[ch2])

    xs = torch.tensor(xs, dtype=torch.long)
    ys = torch.tensor(ys, dtype=torch.long)

    return xs, ys



class BigramNN(nn.Module):
    def __init__(self, vocab_size, seed=seed):
        super().__init__()
        torch.manual_seed(seed)
        self.lookup = nn.Embedding(vocab_size, vocab_size)
        
        nn.init.normal_(self.lookup.weight, mean=0.0, std=1.0)

    def forward(self, x):
        return self.lookup(x)


def train_neural_bigram(
    xs,
    ys,
    vocab_size,
    epochs=200,
    learning_rate=1.0
):
    model = BigramNN(vocab_size, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for epoch in range(epochs):
      
        logits = model(xs)

        loss = F.cross_entropy(logits, ys)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"NLL: {loss.item():.4f}"
            )

    return model


def sample_neural_model(
    model,
    itos,
    num_samples=20,
    seed=seed
):
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    results = []

    with torch.no_grad():
        for _ in range(num_samples):
            result = []
            ix = 0

            while True:
                input_tensor = torch.tensor([ix], dtype=torch.long)
                logits = model(input_tensor)

                # Logit'lerden olasılıklara geçiş
                probs = F.softmax(logits, dim=-1)

                ix = torch.multinomial(
                    probs[0],
                    num_samples=1,
                    replacement=True,
                    generator=generator
                ).item()

                if ix == 0:
                    break

                result.append(itos[ix])

            results.append("".join(result))

    return results


def main():
   
    names = load_names(data_path)

    print("\ndataset")
    print("Number of names:", len(names))
    print("\nFirst 10 names:")
    for name in names[:10]:
        print(repr(name))

    chars, stoi, itos = build_vocabulary(names)
    vocab_size = len(stoi)

    print("\nvocabualry")
   
    print("Vocabulary size:", vocab_size)
    print("Characters:")
    for i in range(vocab_size):
        print(i, repr(itos[i]))

    bigrams = count_bigrams_dictionary(names)

    print("\nBIGRAM DICTIONARY")
   
    print("Most frequent bigrams:")
    for bigram, count in sorted(
        bigrams.items(),
        key=lambda x: x[1],
        reverse=True
    )[:20]:
        print(repr(bigram), count)

    N = build_count_matrix(names, stoi)

    print("Shape:", N.shape)
    print("Total bigrams= ", N.sum().item())

    visualize_matrix(N, itos)

    P = build_probability_matrix(N, smoothing=smoothing)

    print("\nRow sums:")
    print(P.sum(dim=1))

    count_nll = calculate_count_nll(names, stoi, P)

    print("\ncount model")
  
    print("NLL:", count_nll.item())

    print("\nGenerated names:")
    samples = sample_count_model(P, itos, number_of_sumbles)
    for name in samples:
        print(repr(name))

    xs, ys = build_training_data(names, stoi)

    print("X shape:", xs.shape)
    print("Y shape:", ys.shape)

    print("\ntrainng biagram")
   
    model = train_neural_bigram(
        xs,
        ys,
        vocab_size,
        epochs=epoches,
        learning_rate=lr
    )

    print("\nneural model")
    samples = sample_neural_model(model, itos, number_of_sumbles)
    for name in samples:
        print(repr(name))


if __name__ == "__main__":
    main()