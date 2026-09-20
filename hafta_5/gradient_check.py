import torch

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from hafta_4.turkish_names_mlp import load_data, create_model


BLOCK_SIZE = 3
EMBEDDING_DIM = 2
HIDDEN_DIM = 100

BATCH_SIZE = 128

MOMENTUM = 0.1
EPS = 1e-5


def create_bn_parameters():
    gamma = torch.ones(
        HIDDEN_DIM,
        requires_grad=True
    )

    beta = torch.zeros(
        HIDDEN_DIM,
        requires_grad=True
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
    training=True,
    running_mean=None,
    running_var=None,
    retain_grads=False
):

    C, W1, _, W2, b2 = model[:5]

    B = X.shape[0]
    
    emb = C[X]

    emb_flat = emb.view(
        B,
        -1
    )

    z = emb_flat @ W1

   
    if training:

        batch_mean = z.mean(
            dim=0,
            keepdim=True
        )

        batch_var = z.var(
            dim=0,
            keepdim=True,
            unbiased=False
        )

        z_centered = z - batch_mean

        std = torch.sqrt(
            batch_var + EPS
        )

        inv_std = 1.0 / std

        z_norm = z_centered * inv_std

        if running_mean is not None:

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

    else:

        batch_mean = None
        batch_var = None
        z_centered = None

        std = torch.sqrt(
            running_var + EPS
        )

        inv_std = 1.0 / std

        z_norm = (
            z - running_mean
        ) * inv_std

   
    z_bn = gamma * z_norm + beta

    h = torch.tanh(z_bn)
  
    logits = h @ W2 + b2


    #e tabanında üssünü alma
    counts = logits.exp()

    counts_sum = counts.sum(
        dim=1,
        keepdim=True
    )

    probs = counts / counts_sum

    correct_probs = probs[
        torch.arange(B),
        Y
    ]

    logprobs = correct_probs.log()

    loss = -logprobs.mean()


    if retain_grads:

        tensors = [
            emb,
            emb_flat,
            z,
            batch_mean,
            batch_var,
            z_centered,
            std,
            inv_std,
            z_norm,
            z_bn,
            h,
            logits,
            counts,
            counts_sum,
            probs,
            correct_probs,
            logprobs
        ]

        for tensor in tensors:

            if tensor is not None:
                tensor.retain_grad()

    cache = {
        "emb": emb,
        "emb_flat": emb_flat,

        "z": z,

        "batch_mean": batch_mean,
        "batch_var": batch_var,
        "z_centered": z_centered,
        "std": std,
        "inv_std": inv_std,
        "z_norm": z_norm,

        "z_bn": z_bn,
        "h": h,

        "logits": logits,

        "counts": counts,
        "counts_sum": counts_sum,
        "probs": probs,
        "correct_probs": correct_probs,
        "logprobs": logprobs,

        "loss": loss
    }

    return loss, cache



def compare(name, manual, reference):

    exact = torch.equal(
        manual,
        reference
    )

    approximate = torch.allclose(
        manual,
        reference,
        rtol=1e-5,
        atol=1e-6
    )

    max_diff = (
        manual - reference
    ).abs().max().item()

    print(
        f"{name:20s}  "
        f"exact: {str(exact):5s}  "
        f"approx: {str(approximate):5s}  "
        f"max diff: {max_diff:.10f}"
    )



def exercise_1(data):

    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = data

    vocab_size = len(stoi)

    torch.manual_seed(10)

    model = create_model(vocab_size)

    gamma, beta = create_bn_parameters()

    running_mean, running_var = create_running_statistics()

    
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
        training=True,
        running_mean=running_mean,
        running_var=running_var,
        retain_grads=True
    )

    print(f"\nLoss: {loss.item():.6f}")

   
    parameters = list(model[5]) + [
        gamma,
        beta
    ]

    for p in parameters:
        p.grad = None

    loss.backward()

   
    B = X.shape[0]

   
    dlogprobs = (
        -torch.ones_like(
            cache["logprobs"]
        ) / B
    )

    compare(
        "logprobs",
        dlogprobs,
        cache["logprobs"].grad
    )

   
    dcorrect_probs = (
        dlogprobs /
        cache["correct_probs"]
    )

    compare(
        "correct_probs",
        dcorrect_probs,
        cache["correct_probs"].grad
    )

   
    dprobs = torch.zeros_like(
        cache["probs"]
    )

    dprobs[
        torch.arange(B),
        Y
    ] = dcorrect_probs

    compare(
        "probs",
        dprobs,
        cache["probs"].grad
    )

   
    counts = cache["counts"]
    counts_sum = cache["counts_sum"]

    dcounts = (
        dprobs /
        counts_sum
    )

    dcounts_sum = -(
        dprobs * counts
    ).sum(
        dim=1,
        keepdim=True
    ) / (
        counts_sum ** 2
    )

    dcounts += dcounts_sum

    compare(
        "counts_sum",
        dcounts_sum,
        cache["counts_sum"].grad
    )

    compare(
        "counts",
        dcounts,
        cache["counts"].grad
    )

   
    dlogits = (
        dcounts *
        counts
    )

    compare(
        "logits",
        dlogits,
        cache["logits"].grad
    )

    
    h = cache["h"]

    W2 = model[3]

    dW2 = h.T @ dlogits

    db2 = dlogits.sum(
        dim=0
    )

    dh = dlogits @ W2.T

    compare(
        "W2",
        dW2,
        model[3].grad
    )

    compare(
        "b2",
        db2,
        model[4].grad
    )

    compare(
        "h",
        dh,
        cache["h"].grad
    )

   
    dz_bn = (
        1 - h ** 2
    ) * dh

    compare(
        "z_bn",
        dz_bn,
        cache["z_bn"].grad
    )

    
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

    dz_norm = (
        dz_bn * gamma
    )

    compare(
        "gamma",
        dgamma,
        gamma.grad
    )

    compare(
        "beta",
        dbeta,
        beta.grad
    )

    compare(
        "z_norm",
        dz_norm,
        cache["z_norm"].grad
    )

   
    # z_norm = z_centered * inv_std
   

    z_centered = cache["z_centered"]
    inv_std = cache["inv_std"]
    batch_var = cache["batch_var"]

    dz_centered = (
        dz_norm *
        inv_std
    )

    dinv_std = (
        dz_norm *
        z_centered
    ).sum(
        dim=0,
        keepdim=True
    )

    compare(
        "z_centered",
        dz_centered,
        cache["z_centered"].grad
    )

    compare(
        "inv_std",
        dinv_std,
        cache["inv_std"].grad
    )

    # inv_std = (batch_var + EPS)^(-1/2)

    dvar = (
        dinv_std *
        (-0.5) *
        (batch_var + EPS) ** (-1.5)
    )

    compare(
        "batch_var",
        dvar,
        cache["batch_var"].grad
    )

    # batch_var = mean(z_centered^2)
   
    dz_centered_from_var = (
        dvar *
        2 *
        z_centered /
        B
    )

    dz_from_centered = (
        dz_centered +
        dz_centered_from_var
    )

    # z_centered = z - batch_mean

    dbatch_mean = -dz_from_centered.sum(
        dim=0,
        keepdim=True
    )

    dz = dz_from_centered.clone()

    # batch_mean = mean(z)

    dz += dbatch_mean / B

    compare(
        "batch_mean",
        dbatch_mean,
        cache["batch_mean"].grad
    )

    compare(
        "z",
        dz,
        cache["z"].grad
    )

    # z = emb_flat @ W1

    emb_flat = cache["emb_flat"]

    W1 = model[1]

    dW1 = emb_flat.T @ dz

    demb_flat = dz @ W1.T

    compare(
        "W1",
        dW1,
        model[1].grad
    )

    compare(
        "emb_flat",
        demb_flat,
        cache["emb_flat"].grad
    )

    # flatten
    demb = demb_flat.view_as(
        cache["emb"]
    )

    compare(
        "emb",
        demb,
        cache["emb"].grad
    )
    C = model[0]

    dC = torch.zeros_like(C)

    dC.index_add_(
        0,
        X.reshape(-1),
        demb.reshape(
            -1,
            EMBEDDING_DIM
        )
    )

    compare(
        "C",
        dC,
        C.grad
    )

  


def exercise_2(data):
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = data

    vocab_size = len(stoi)

    torch.manual_seed(10)

    model = create_model(vocab_size)

    gamma, beta = create_bn_parameters()

    running_mean, running_var = create_running_statistics()

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
        training=True,
        running_mean=running_mean,
        running_var=running_var,
        retain_grads=True
    )

    parameters = list(model[5]) + [
        gamma,
        beta
    ]

    for p in parameters:
        p.grad = None

    loss.backward()

    B = X.shape[0]

   
    dlogits = cache["probs"].clone()

    dlogits[
        torch.arange(B),
        Y
    ] -= 1

    dlogits /= B

    compare(
        "compact dlogits",
        dlogits,
        cache["logits"].grad
    )

    



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
        x_centered * inv_std
    )

    dxhat = dout * gamma

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

    dgamma = (
        dout * x_hat
    ).sum(
        dim=0
    )

    dbeta = dout.sum(
        dim=0
    )

    return dx, dgamma, dbeta


def exercise_3(data):


    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = data

    vocab_size = len(stoi)

    torch.manual_seed(10)

    model = create_model(vocab_size)

    gamma, beta = create_bn_parameters()

    running_mean, running_var = create_running_statistics()

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
        training=True,
        running_mean=running_mean,
        running_var=running_var,
        retain_grads=True
    )

    parameters = list(model[5]) + [
        gamma,
        beta
    ]

    for p in parameters:
        p.grad = None

    loss.backward()

    dout = cache["z_bn"].grad

    dz, dgamma, dbeta = batchnorm_backward(
        dout,
        cache["z"],
        gamma,
        cache["batch_mean"],
        cache["batch_var"]
    )

    compare(
        "compact BN dz",
        dz,
        cache["z"].grad
    )

    compare(
        "compact BN dgamma",
        dgamma,
        gamma.grad
    )

    compare(
        "compact BN dbeta",
        dbeta,
        beta.grad
    )

  


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


    print("\nTrain:", Xtr.shape)
    print("Dev:  ", Xdev.shape)
    print("Test: ", Xte.shape)

    print(
        "Vocabulary size:",
        len(stoi)
    )

    print(
        "Characters:",
        list(stoi.keys())
    )

    exercise_1(data)
    exercise_2(data)
    exercise_3(data)


if __name__ == "__main__":
    main()