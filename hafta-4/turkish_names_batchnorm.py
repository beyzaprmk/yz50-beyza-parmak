import torch
import torch.nn.functional as F

from turkish_names_mlp import load_data, create_model


# z = w_1 x_1 + w_2 x_2 + b

BATCH_SIZE = 32
STEPS = 10000
LEARNING_RATE = 0.2
MOMENTUM = 0.1
EPS = 1e-5


def forward_bn(X, Y, model, training, running_mean, running_var, gamma, beta):
    C, W1, _, W2, b2 = model[:5]

    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)

    z = emb_flat @ W1

    # Batch Normalization step
    if training:
        batch_mean = z.mean(dim=0, keepdim=True)       # shape: (1, 100)
        batch_var = z.var(dim=0, keepdim=True, unbiased=False) 

        z_norm = (z - batch_mean) / torch.sqrt(batch_var + EPS)

        with torch.no_grad():
            running_mean.mul_(1 - MOMENTUM).add_(MOMENTUM * batch_mean)
            running_var.mul_(1 - MOMENTUM).add_(MOMENTUM * batch_var)
    else:
        z_norm = (z - running_mean) / torch.sqrt(running_var + EPS)

    z_bn = gamma * z_norm + beta

    h = torch.tanh(z_bn)

    logits = h @ W2 + b2

    return F.cross_entropy(logits, Y)


def train_bn(Xtr, Ytr, Xdev, Ydev, vocab_size):
    model = create_model(vocab_size)
    gamma = torch.ones(100, requires_grad=True)
    beta = torch.zeros(100, requires_grad=True)

    running_mean = torch.zeros((1, 100))
    running_var = torch.ones((1, 100))

    parameters = list(model[5]) + [gamma, beta]

    for step in range(STEPS):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        loss = forward_bn(
            Xtr[ix], Ytr[ix], model, True,
            running_mean, running_var, gamma, beta
        )

        for p in parameters:
            p.grad = None

        loss.backward()

        with torch.no_grad():
            for p in parameters:
                if p.grad is not None:
                    p -= LEARNING_RATE * p.grad

    dev_loss = forward_bn(
        Xdev, Ydev, model, False,
        running_mean, running_var, gamma, beta
    ).item()

    return dev_loss


def train_no_bn(Xtr, Ytr, Xdev, Ydev, vocab_size):
    model = create_model(vocab_size)

    for step in range(STEPS):
        ix = torch.randint(0, Xtr.shape[0], (BATCH_SIZE,))
        loss = model_loss(Xtr[ix], Ytr[ix], model)

        for p in model[5]:
            p.grad = None

        loss.backward()

        with torch.no_grad():
            for p in model[5]:
                if p.grad is not None:
                    p -= LEARNING_RATE * p.grad

    return model_loss(Xdev, Ydev, model).item()


def model_loss(X, Y, model):
    C, W1, b1, W2, b2 = model[:5]
    emb = C[X]
    emb_flat = emb.view(emb.shape[0], -1)
    h = torch.tanh(emb_flat @ W1 + b1)
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y)


def main():
    Xtr, Ytr, Xdev, Ydev, *_rest, stoi, itos = load_data()
    vocab_size = len(stoi)

    no_bn_loss = train_no_bn(Xtr, Ytr, Xdev, Ydev, vocab_size)
    bn_loss = train_bn(Xtr, Ytr, Xdev, Ydev, vocab_size)

    print(f"No BatchNorm dev loss: {no_bn_loss:.4f}")
    print(f"BatchNorm dev loss:    {bn_loss:.4f}")


if __name__ == "__main__":
    main()
