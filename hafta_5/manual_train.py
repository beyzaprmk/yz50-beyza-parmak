import torch

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hafta_4.turkish_names_mlp import load_data, create_model


BLOCK_SIZE = 3
EMBEDDING_DIM = 2
HIDDEN_DIM = 100

BATCH_SIZE = 64

STEPS = 10000
LEARNING_RATE = 0.2

MOMENTUM = 0.1
EPS = 1e-5

def create_bn_parameters():

    gamma = torch.ones(
        HIDDEN_DIM,
        requires_grad=False
    )

    beta = torch.zeros(
        HIDDEN_DIM,
        requires_grad=False
    )

    return gamma, beta


def create_running_statistics():

    running_mean = torch.zeros(
        1,
        HIDDEN_DIM
    )

    running_var = torch.ones(
        1,
        HIDDEN_DIM
    )

    return running_mean, running_var


def forward_bn(
    X,
    Y,
    model,
    gamma,
    beta,
    running_mean,
    running_var
):

    C, W1, _, W2, b2 = model[:5]

    B = X.shape[0]

    emb = C[X]

    emb_flat = emb.view(
        B,
        -1
    )
    z = emb_flat @ W1

    
    batch_mean = z.mean(
        dim=0,
        keepdim=True
    )

    batch_var = z.var(
        dim=0,
        keepdim=True,
        unbiased=False
    )

    z_centered = (
        z - batch_mean
    )

    inv_std = 1.0 / torch.sqrt(
        batch_var + EPS
    )

    z_norm = (
        z_centered * inv_std
    )

    with torch.no_grad():

        running_mean.mul_(
            1 - MOMENTUM
        ).add_(
            MOMENTUM * batch_mean
        )

        running_var.mul_(
            1 - MOMENTUM
        ).add_(
            MOMENTUM * batch_var
        )

    z_bn = (
        gamma * z_norm +
        beta
    )

    h = torch.tanh(
        z_bn
    )
    logits = (
        h @ W2 +
        b2
    )
    counts = logits.exp()

    probs = (
        counts /
        counts.sum(
            dim=1,
            keepdim=True
        )
    )

    correct_probs = probs[
        torch.arange(B),
        Y
    ]

    logprobs = correct_probs.log()

    loss = -logprobs.mean()

    cache = {
        "emb": emb,
        "emb_flat": emb_flat,

        "z": z,

        "batch_mean": batch_mean,
        "batch_var": batch_var,
        "z_centered": z_centered,
        "inv_std": inv_std,
        "z_norm": z_norm,

        "z_bn": z_bn,
        "h": h,

        "logits": logits,
        "probs": probs
    }

    return loss, cache

def batchnorm_backward(
    dout,
    x,
    gamma,
    batch_mean,
    batch_var
):

    B = x.shape[0]

    x_centered = (
        x - batch_mean
    )

    inv_std = 1.0 / torch.sqrt(
        batch_var + EPS
    )

    x_hat = (
        x_centered *
        inv_std
    )
    dgamma = (
        dout * x_hat
    ).sum(
        dim=0
    )

    # Gradient beta
    dbeta = dout.sum(
        dim=0
    )

    # Gradient to normalized x
    dxhat = (
        dout * gamma
    )

    dx = (
        inv_std / B
    ) * (
        B * dxhat
        - dxhat.sum(
            dim=0,
            keepdim=True
        )
        - x_hat * (
            dxhat * x_hat
        ).sum(
            dim=0,
            keepdim=True
        )
    )

    return dx, dgamma, dbeta

def manual_backward(
    X,
    Y,
    model,
    gamma,
    beta,
    cache
):

    B = X.shape[0]

    C, W1, b1, W2, b2 = model[:5]


    probs = cache["probs"]

    dlogits = probs.clone()

    dlogits[
        torch.arange(B),
        Y
    ] -= 1

    dlogits /= B

    h = cache["h"]

    dW2 = h.T @ dlogits

    db2 = dlogits.sum(
        dim=0
    )

    dh = dlogits @ W2.T
    dz_bn = (
        1 - h ** 2
    ) * dh

    # z_bn = gamma * z_norm + beta

    z_norm = cache["z_norm"]

    dgamma = (
        dz_bn * z_norm
    ).sum(
        dim=0
    )

    dbeta = dz_bn.sum(
        dim=0
    )

    dz, _, _ = batchnorm_backward(
        dz_bn,
        cache["z"],
        gamma,
        cache["batch_mean"],
        cache["batch_var"]
    )

    emb_flat = cache["emb_flat"]

    dW1 = emb_flat.T @ dz

    demb_flat = dz @ W1.T

    demb = demb_flat.view_as(
        cache["emb"]
    )

    dC = torch.zeros_like(
        C
    )

    dC.index_add_(
        0,
        X.reshape(-1),
        demb.reshape(
            -1,
            EMBEDDING_DIM
        )
    )

    
    db1 = torch.zeros_like(
        b1
    )

    return {
        "C": dC,
        "W1": dW1,
        "b1": db1,
        "W2": dW2,
        "b2": db2,
        "gamma": dgamma,
        "beta": dbeta
    }

def evaluate(
    X,
    Y,
    model,
    gamma,
    beta,
    running_mean,
    running_var
):

    C, W1, _, W2, b2 = model[:5]

    B = X.shape[0]

    emb = C[X]

    emb_flat = emb.view(
        B,
        -1
    )

    z = emb_flat @ W1

    z_norm = (
        z - running_mean
    ) / torch.sqrt(
        running_var + EPS
    )

    z_bn = (
        gamma * z_norm +
        beta
    )

    h = torch.tanh(
        z_bn
    )

    logits = (
        h @ W2 +
        b2
    )

    counts = logits.exp()

    probs = (
        counts /
        counts.sum(
            dim=1,
            keepdim=True
        )
    )

    correct_probs = probs[
        torch.arange(B),
        Y
    ]

    loss = -correct_probs.log().mean()

    return loss.item()


def train_manual(
    Xtr,
    Ytr,
    Xdev,
    Ydev,
    Xte,
    Yte,
    vocab_size
):

    torch.manual_seed(10)

    model = create_model(
        vocab_size
    )

    gamma, beta = create_bn_parameters()

    running_mean, running_var = (
        create_running_statistics()
    )

    for step in range(STEPS):
        ix = torch.randint(
            0,
            Xtr.shape[0],
            (BATCH_SIZE,)
        )

        X = Xtr[ix]
        Y = Ytr[ix]

        loss, cache = forward_bn(
            X,
            Y,
            model,
            gamma,
            beta,
            running_mean,
            running_var
        )

        grads = manual_backward(
            X,
            Y,
            model,
            gamma,
            beta,
            cache
        )

        with torch.no_grad():

            # C
            model[0].sub_(
                LEARNING_RATE * grads["C"]
            )

            # W1
            model[1].sub_(
                LEARNING_RATE * grads["W1"]
            )

            model[3].sub_(
                LEARNING_RATE * grads["W2"]
            )

            # b2
            model[4].sub_(
                LEARNING_RATE * grads["b2"]
            )

            # BatchNorm gamma
            gamma.sub_(
                LEARNING_RATE * grads["gamma"]
            )


            # BatchNorm beta
            beta.sub_(
                LEARNING_RATE * grads["beta"]
            )

    
        if step % 1000 == 0:

            print(
                f"step {step:5d}  "
                f"loss {loss.item():.4f}"
            )

    train_loss = evaluate(
        Xtr,
        Ytr,
        model,
        gamma,
        beta,
        running_mean,
        running_var
    )

    dev_loss = evaluate(
        Xdev,
        Ydev,
        model,
        gamma,
        beta,
        running_mean,
        running_var
    )

    test_loss = evaluate(
        Xte,
        Yte,
        model,
        gamma,
        beta,
        running_mean,
        running_var
    )


    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Dev loss:   {dev_loss:.4f}"
    )

    print(
        f"Test loss:  {test_loss:.4f}"
    )

    return (
        model,
        gamma,
        beta,
        running_mean,
        running_var
    )


def generate_names(
    model,
    gamma,
    beta,
    running_mean,
    running_var,
    stoi,
    itos,
    number=20,
    max_length=20
):

    C, W1, _, W2, b2 = model[:5]

   
   
    print("turkish names:")
   

    with torch.no_grad():

        for _ in range(number):

            context = [
                0
            ] * BLOCK_SIZE

            name = ""

            for _ in range(max_length):

                X = torch.tensor(
                    [context],
                    dtype=torch.long
                )

                
                emb = C[X]

                emb_flat = emb.view(
                    1,
                    -1
                )

                z = emb_flat @ W1

                z_norm = (
                    z - running_mean
                ) / torch.sqrt(
                    running_var + EPS
                )

                z_bn = (
                    gamma * z_norm +
                    beta
                )
                h = torch.tanh(
                    z_bn
                )

                
                logits = (
                    h @ W2 +
                    b2
                )

                probs = torch.softmax(
                    logits,
                    dim=1
                )

                ix = torch.multinomial(
                    probs,
                    1
                ).item()

                # End token
                if ix == 0:
                    break

                name += itos[ix]

                context = (
                    context[1:]
                    + [ix]
                )

            print(name)


def main():
    data = load_data()

    (
        Xtr,
        Ytr,
        Xdev,
        Ydev,
        Xte,
        Yte,
        stoi,
        itos
    ) = data

    vocab_size = len(stoi)


    print("\nDataset:")

    print(
        "Train:",
        Xtr.shape
    )

    print(
        "Dev:  ",
        Xdev.shape
    )

    print(
        "Test: ",
        Xte.shape
    )

    print(
        "Vocabulary:",
        vocab_size
    )

   
    (
        model,
        gamma,
        beta,
        running_mean,
        running_var
    ) = train_manual(
        Xtr,
        Ytr,
        Xdev,
        Ydev,
        Xte,
        Yte,
        vocab_size
    )

    
    generate_names(
        model,
        gamma,
        beta,
        running_mean,
        running_var,
        stoi,
        itos
    )


if __name__ == "__main__":
    main()