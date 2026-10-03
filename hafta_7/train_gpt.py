import torch

from data_loader import load_data, get_batch
from model import GPTLanguageModel


BATCH_SIZE = 64
BLOCK_SIZE = 64

N_EMBD = 64
LEARNING_RATE = 3e-4
STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 100

SEED = 42


@torch.no_grad()
def estimate_loss( model, train_data,val_data):
    model.eval()
    losses = {}

    for split in ["train", "val"]:

        total_loss = 0.0

        for _ in range(EVAL_ITERS):

            X, Y = get_batch( split, BATCH_SIZE,BLOCK_SIZE,train_data,val_data)

            _, loss = model(X, Y)

            total_loss += loss.item()

        losses[split] = total_loss / EVAL_ITERS

    model.train()

    return losses


def train(n_head):

    torch.manual_seed(SEED)

    train_data, val_data, tokenizer = load_data()

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=n_head
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    print(f"Training model with {n_head} attention head(s)")

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    for step in range(STEPS):

        X, Y = get_batch("train",BATCH_SIZE,BLOCK_SIZE,train_data,val_data )

        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)

        loss.backward()

        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == STEPS - 1:

            losses = estimate_loss(model,train_data,val_data )  

            print(
                f"step {step:4d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

    final_losses = estimate_loss(model, train_data,val_data)
  
    print(f"Final Train loss: {final_losses['train']:.4f}")
    print(f"Final Val loss:   {final_losses['val']:.4f}")

    torch.save(
    model.state_dict(),
    f"outputs/model_{n_head}head.pt")

    return final_losses


if __name__ == "__main__":
    single_head = train(n_head=1)
    multi_head = train(n_head=4)

    print(
        f"Single Head  | "
        f"train: {single_head['train']:.4f} | "
        f"val: {single_head['val']:.4f}"
    )
    print(
        f"Multi Head   | "
        f"train: {multi_head['train']:.4f} | "
        f"val: {multi_head['val']:.4f}"
    )