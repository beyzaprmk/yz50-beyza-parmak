import torch
import torch.nn.functional as F

from data import load_data
from model import (
    MLP,
    WaveNet,
    WaveNetBuggyBN,
    count_parameters
)


BATCH_SIZE = 64
STEPS = 10000
LEARNING_RATE = 0.1

WEEK4_DEV_LOSS = 1.9637

def loss_fn(logits, Y):
    return F.cross_entropy(logits, Y)


def prepare_logits(logits):
    
    if logits.ndim == 3:
        logits = logits[:, 0, :]

    return logits


def evaluate(model, X, Y):

    model.eval()

    with torch.no_grad():

        logits = model(X)
        logits = prepare_logits(logits)

        loss = loss_fn(
            logits,
            Y
        )

    model.train()

    return loss.item()

def train(
    model,
    Xtr,
    Ytr,
    steps=STEPS,
    learning_rate=LEARNING_RATE,
    batch_size=BATCH_SIZE
):

    parameters = model.parameters()

    model.train()

    for step in range(steps):

        ix = torch.randint(0,Xtr.shape[0],(batch_size,)
        )

        X = Xtr[ix]
        Y = Ytr[ix]

        logits = model(X)
        logits = prepare_logits(logits)

        loss = loss_fn(
            logits,
            Y
        )

        for p in parameters:
            p.grad = None

        loss.backward()

        
        with torch.no_grad():

            for p in parameters:
                p -= learning_rate * p.grad

        if step % 1000 == 0:

            print(
                f"step {step:5d} "
                f"loss {loss.item():.4f}"
            )


def run_experiment(
    model,
    Xtr,
    Ytr,
    Xdev,
    Ydev,
    Xte,
    Yte,
    name
):

    print()
    print(name)

    parameter_count = count_parameters(
        model
    )

    print(
        f"Parameters: {parameter_count}"
    )

    train(model,Xtr, Ytr
    )

    train_loss = evaluate(model, Xtr, Ytr
    )

    dev_loss = evaluate( model, Xdev, Ydev
    )

    test_loss = evaluate( model, Xte,  Yte
    )

    print()
    print(
        f"Train loss: {train_loss:.4f}"
    )

    print(
        f"Dev loss:   {dev_loss:.4f}"
    )

    print(
        f"Test loss:  {test_loss:.4f}"
    )

    return {
        "name": name,
        "parameters": parameter_count,
        "train_loss": train_loss,
        "dev_loss": dev_loss,
        "test_loss": test_loss,
        "model": model
    }


def compare_models(results):

    print(
        f"{'Model':30s}"
        f"{'Parameters':>15s}"
        f"{'Dev Loss':>15s}"
    )

    for result in results:

        print(
            f"{result['name']:30s}"
            f"{result['parameters']:>15d}"
            f"{result['dev_loss']:>15.4f}"
        )

def compare_batchnorm(
    buggy_result,
    correct_result
):

    buggy_loss = buggy_result["dev_loss"]
    correct_loss = correct_result["dev_loss"]

    difference = buggy_loss - correct_loss

    print(
        f"Incorrect 3D BN Dev Los : "
        f"{buggy_loss:.4f}"
    )

    print(
        f"Correct 3D BN Dev Loss: "
        f"{correct_loss:.4f}"
    )

    print(
        f"Dev Loss Difference      : "
        f"{difference:+.4f}"
    )

def compare_with_week4(
    week6_wavenet_dev_loss
):

    difference = (
        WEEK4_DEV_LOSS
        - week6_wavenet_dev_loss
    )


    print(
        f"{WEEK4_DEV_LOSS:.4f}"
    )

    print(
        f"{week6_wavenet_dev_loss:.4f}"
    )

    print(
        f"Dev Loss Difference: "
        f"{difference:+.4f}"
    )

    print()
    

def sample_names(
    model,
    itos,
    num_names=10,
    block_size=8,
    max_length=20
):

    model.eval()

    generated_names = []

    with torch.no_grad():

        for _ in range(num_names):

            context = [0] * block_size

            name = []

            for _ in range(max_length):

                X = torch.tensor(
                    [context],
                    dtype=torch.long
                )

                logits = model(X)

                logits = prepare_logits(
                    logits
                )

                probabilities = F.softmax(
                    logits,
                    dim=1
                )

                ix = torch.multinomial(
                    probabilities,
                    num_samples=1
                ).item()

                # "." = end of name
                if ix == 0:
                    break

                name.append(
                    itos[ix]
                )

                context = (
                    context[1:]
                    + [ix]
                )

            generated_names.append(
                "".join(name)
            )

    model.train()

    return generated_names


def print_samples(
    model,
    itos,
    num_names=10
):

    names = sample_names(
        model,
        itos,
        num_names=num_names
    )

    print()
    print("TURKISH NAMES")

    for i, name in enumerate(names, 1):

        print(
            f"{i:2d}. {name}"
        )

if __name__ == "__main__":

    (
        Xtr3,
        Ytr3,
        Xdev3,
        Ydev3,
        Xte3,
        Yte3,
        vocab_size,
        stoi,
        itos
    ) = load_data(
        block_size=3
    )

    print()
    print("Context 3")

    print(
        f"Train: {Xtr3.shape}"
    )
    print(
        f"Dev:   {Xdev3.shape}"
    )
    print(
        f"Test:  {Xte3.shape}"
    )
    print(
        f"Vocab: {vocab_size}"
    )


    model_mlp_3 = MLP(
        vocab_size=vocab_size,
        block_size=3,
        embedding_dim=2,
        hidden_dim=100
    )

    result_mlp_3 = run_experiment( model_mlp_3,Xtr3,Ytr3,  Xdev3,  Ydev3,  Xte3,  Yte3,  "Context 3 MLP"
    )

    (
        Xtr8,
        Ytr8,
        Xdev8,
        Ydev8,
        Xte8,
        Yte8,
        vocab_size,
        stoi,
        itos
    ) = load_data(
        block_size=8
    )

    print()
    print("Context 8")
    print(
        f"Train: {Xtr8.shape}"
    )
    print(
        f"Dev:   {Xdev8.shape}"
    )
    print(
        f"Test:  {Xte8.shape}"
    )
    print(
        f"Vocab: {vocab_size}"
    )
    model_mlp_8 = MLP(
        vocab_size=vocab_size,
        block_size=8,
        embedding_dim=2,
        hidden_dim=100
    )

    result_mlp_8 = run_experiment(
        model_mlp_8,
        Xtr8,
        Ytr8,
        Xdev8,
        Ydev8,
        Xte8,
        Yte8,
        "Context 8 Flat MLP"
    )

    model_wavenet = WaveNet(
        vocab_size=vocab_size,
        block_size=8,
        embedding_dim=2,
        hidden_dim=100
    )

    result_wavenet = run_experiment(
        model_wavenet,
        Xtr8,
        Ytr8,
        Xdev8,
        Ydev8,
        Xte8,
        Yte8,
        "Context 8 WaveNet - Correct BN"
    )

    model_wavenet_buggy = WaveNetBuggyBN(
        vocab_size=vocab_size,
        block_size=8,
        embedding_dim=2,
        hidden_dim=100
    )

    result_wavenet_buggy = run_experiment(
        model_wavenet_buggy,
        Xtr8,
        Ytr8,
        Xdev8,
        Ydev8,
        Xte8,
        Yte8,
        "Context 8 WaveNet - Buggy 3D BN"
    )
    compare_batchnorm(
        result_wavenet_buggy,
        result_wavenet
    )

    model_large = WaveNet(
        vocab_size=vocab_size,
        block_size=8,
        embedding_dim=10,
        hidden_dim=200
    )

    result_large = run_experiment(
        model_large,
        Xtr8,
        Ytr8,
        Xdev8,
        Ydev8,
        Xte8,
        Yte8,
        "Context 8 WaveNet Large"
    )

    compare_models([
        result_mlp_3,
        result_mlp_8,
        result_wavenet
    ])

    compare_with_week4(
        result_wavenet["dev_loss"]
    )

    print_samples(
        result_wavenet["model"],
        itos,
        num_names=10
    )