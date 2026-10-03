import os

import torch
import matplotlib.pyplot as plt

from data_loader import load_data, get_batch
from model import GPTLanguageModel


BLOCK_SIZE = 64
N_EMBD = 64
BATCH_SIZE = 1

OUTPUT_DIR = "outputs/attention"


def plot_attention(
    model,
    idx,
    tokenizer,
    head_number=0,
    max_tokens=16
):
    model.eval()

    with torch.no_grad():
        model(idx)

    weights = model.get_attention_weights()

    matrix = weights[head_number][0]

    T = min(max_tokens, matrix.shape[0])

    matrix = matrix[:T, :T]

    token_ids = idx[0, :T].tolist()

    labels = [
        tokenizer.decode([token])
        for token in token_ids
    ]

    plt.figure(figsize=(8, 8))

    plt.imshow(
        matrix.cpu(),
        aspect="auto"
    )

    plt.xticks(
        range(T),
        labels
    )

    plt.yticks(
        range(T),
        labels
    )

    plt.xlabel("Key")
    plt.ylabel("Query")

    plt.title(
        f"Attention Head {head_number + 1}"
    )

    plt.colorbar()

    output_path = os.path.join(
        OUTPUT_DIR,
        f"head_{head_number + 1}.png"
    )

    plt.savefig(
        output_path,
        bbox_inches="tight"
    )

    plt.close()


def main():
    train_data, val_data, tokenizer = load_data()

    # Output klasörünü oluştur
    os.makedirs(OUTPUT_DIR,exist_ok=True
    )

    # 4-head model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=4
    )

    model.load_state_dict(
        torch.load(
            "outputs/model_4head.pt",
            weights_only=True
        )
    )

    # Validation'dan bir örnek
    X, Y = get_batch(
        "val",
        BATCH_SIZE,
        BLOCK_SIZE,
        train_data,
        val_data
    )

    for head_number in range(4):

        plot_attention(model,X,tokenizer, head_number=head_number,max_tokens=16
        )


if __name__ == "__main__":
    main()