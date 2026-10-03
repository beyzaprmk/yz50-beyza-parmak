import torch

from data_loader import load_data, get_batch
from bigram import BigramLanguageModel


# -----------------------------
# Hyperparameters
# -----------------------------

BATCH_SIZE = 64
BLOCK_SIZE = 8

LEARNING_RATE = 1e-2
STEPS = 5000

EVAL_INTERVAL = 500

SEED = 42

@torch.no_grad()
def estimate_loss(
    model,
    train_data,
    val_data,
    eval_iters=200
):
    model.eval()

    losses = {}

    for split in ["train", "val"]:

        total_loss = 0.0

        for _ in range(eval_iters):

            X, Y = get_batch(split, BATCH_SIZE,BLOCK_SIZE,train_data,val_data)

            _, loss = model(X, Y)

            total_loss += loss.item()

        losses[split] = total_loss / eval_iters

    model.train()

    return losses


def train():

    torch.manual_seed(SEED)

    # Data
    train_data, val_data, tokenizer = load_data()

    vocab_size = tokenizer.vocab_size

    print("Vocabulary size:", vocab_size)
    print("Train tokens:", len(train_data))
    print("Val tokens:", len(val_data))

    # Model
    model = BigramLanguageModel(
        vocab_size
    )

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

    for step in range(STEPS):

        # Random batch
        X, Y = get_batch( "train",BATCH_SIZE,BLOCK_SIZE,train_data,val_data)

        # Forward
        logits, loss = model(X, Y)

        # Backward
        optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        optimizer.step()

        # Evaluation
        if step % EVAL_INTERVAL == 0 or step == STEPS - 1:

            losses = estimate_loss( model, train_data, val_data)

            print(f"step {step:5d} | "
                f"train loss {losses['train']:.4f} | "
                f"val loss {losses['val']:.4f}"
            )

    # Final result
    losses = estimate_loss(model,train_data,val_data)
  
    print(f"Train loss: {losses['train']:.4f}"
    )
    print(f"Val loss:   {losses['val']:.4f}"
    )

    torch.save(
        model.state_dict(),
        f"outputs/bigram.pt")
    return model, tokenizer


if __name__ == "__main__":
    train()