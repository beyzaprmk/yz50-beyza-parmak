import torch
from data_loader import load_data, get_batch
from model import GPTLanguageModel

BATCH_SIZE = 64
BLOCK_SIZE = 64
N_EMBD = 64
N_HEAD = 4
LEARNING_RATE = 3e-4
STEPS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 100
SEED = 42

@torch.no_grad()
def estimate_loss(model,train_data,val_data):
    model.eval()
    losses = {}

    for split in ["train", "val"]:

        total_loss = 0.0

        for _ in range(EVAL_ITERS):

            X, Y = get_batch(split, BATCH_SIZE,BLOCK_SIZE,train_data,val_data)

            _, loss = model(X, Y)

            total_loss += loss.item()

        losses[split] = total_loss / EVAL_ITERS

    model.train()

    return losses


def train():

    torch.manual_seed(SEED)

    train_data, val_data, tokenizer = load_data()

    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        causal=False
    )

    optimizer = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE )

    print(
        "Parameters:",
        sum(p.numel() for p in model.parameters())
    )

    for step in range(STEPS):

        X, Y = get_batch("train",BATCH_SIZE,BLOCK_SIZE,train_data,val_data)

        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True )

        loss.backward()
        optimizer.step()

        if step % EVAL_INTERVAL == 0 or step == STEPS - 1:

            losses = estimate_loss(model,train_data, val_data)

            print(
                f"step {step:4d} | "
                f"train {losses['train']:.4f} | "
                f"val {losses['val']:.4f}"
            )

    losses = estimate_loss( model,train_data,val_data)

    print(
        f"Final Train loss: {losses['train']:.4f}"
    )
    print(  f"Final Val loss: {losses['val']:.4f}" )

    torch.save(
        model.state_dict(),
        "outputs/model_4head_noncausal.pt"
    )

if __name__ == "__main__":
    train()