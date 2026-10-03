import torch
import matplotlib.pyplot as plt
from data_loader import load_data, get_batch
from bigram import BigramLanguageModel
from model import GPTLanguageModel

BATCH_SIZE = 64
BLOCK_SIZE = 64
N_EMBD = 64
N_HEAD = 4
EVAL_ITERS = 200
OUTPUT_PATH = "outputs/loss_comparison.png"
 
@torch.no_grad()
def evaluate_model(model,train_data,val_data):
    model.eval()

    total_loss = 0.0

    for _ in range(EVAL_ITERS):

        X, Y = get_batch("val",BATCH_SIZE,BLOCK_SIZE,train_data,val_data
        )

        _, loss = model(X, Y)

        total_loss += loss.item()

    model.train()

    return total_loss / EVAL_ITERS


def load_bigram(
    vocab_size
):

    model = BigramLanguageModel(
        vocab_size
    )

    model.load_state_dict(
        torch.load(
            "outputs/bigram.pt",
            weights_only=True
        )
    )

    return model

def load_gpt( vocab_size,n_head,causal):

    model = GPTLanguageModel( vocab_size=vocab_size, block_size=BLOCK_SIZE, n_embd=N_EMBD,n_head=n_head,
        causal=causal
    )

    if causal:
        checkpoint = "outputs/model_4head.pt"
    else:
        checkpoint = "outputs/model_4head_noncausal.pt"

    model.load_state_dict(
        torch.load(checkpoint,weights_only=True)
    )

    return model


def main():

    train_data, val_data, tokenizer = load_data()

    vocab_size = tokenizer.vocab_size

    bigram = load_bigram( vocab_size )

    bigram_loss = evaluate_model(bigram,train_data,val_data)

    single_head = GPTLanguageModel(vocab_size=vocab_size, block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=1,
        causal=True
    )

    single_head.load_state_dict(
        torch.load(
            "outputs/model_1head.pt",
            weights_only=True
        )
    )

    single_head_loss = evaluate_model(single_head,train_data, val_data )

    multi_head = GPTLanguageModel(vocab_size=vocab_size,block_size=BLOCK_SIZE,n_embd=N_EMBD,n_head=N_HEAD,
        causal=True
    )

    multi_head.load_state_dict(
        torch.load(
            "outputs/model_4head.pt",
            weights_only=True
        )
    )

    multi_head_loss = evaluate_model(multi_head,train_data,val_data)

    noncausal_model = GPTLanguageModel(
        vocab_size=vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        causal=False
    )

    noncausal_model.load_state_dict(
        torch.load(
            "outputs/model_4head_noncausal.pt",
            weights_only=True
        )
    )

    noncausal_loss = evaluate_model( noncausal_model,train_data,val_data)

    results = {
        "Bigram": bigram_loss,
        "Single Head": single_head_loss,
        "Multi Head": multi_head_loss,
        "Multi Head\nNo Mask": noncausal_loss
    }

    print(  f"{'Model':<25} | {'Validation Loss':>15}" )

    for name, loss in results.items():

        print(
            f"{name:<25} | {loss:>15.4f}"
        )

    names = list(results.keys())
    losses = list(results.values())

    plt.figure(figsize=(9, 5))

    plt.bar(
        names,
        losses
    )

    plt.ylabel("Validation Loss")
    plt.title("Week 7 Model Comparison")

    plt.tight_layout()

    plt.savefig(OUTPUT_PATH,dpi=150)

    plt.close()
if __name__ == "__main__":
    main()