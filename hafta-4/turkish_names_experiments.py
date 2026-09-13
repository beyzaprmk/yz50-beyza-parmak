from pathlib import Path
import torch

from turkish_names_mlp import load_data, create_model, train, evaluate



def compare_dimensions():
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = load_data()
    vocab_size = len(stoi)

    settings = [
        (2, 50),
        (2, 100),
        (2, 200),
        (2,300),
        (4, 100),
        (8, 100),
        (16,100)
    ]

    results = []



    for embedding_dim, hidden_dim in settings:
        model = create_model(vocab_size, embedding_dim, hidden_dim)
        train(model, Xtr, Ytr, steps=10000, learning_rate=0.2)
        dev_loss = evaluate(Xdev, Ydev, model)

        results.append((embedding_dim, hidden_dim, dev_loss))
        print(
            f"embedding_dim={embedding_dim:<2} "
            f"hidden_dim={hidden_dim:<3} "
            f"dev_loss={dev_loss:.4f}"
        )

    return results



def learning_rate_sweep():
    Xtr, Ytr, Xdev, Ydev, Xte, Yte, stoi, itos = load_data()
    vocab_size = len(stoi)

    learning_rates = [ 0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.3, 0.4]
    results = []

   

    for lr in learning_rates:
        torch.manual_seed(42)
        model = create_model(vocab_size)
        train(model, Xtr, Ytr, steps=10000, learning_rate=lr)

        train_loss = evaluate(Xtr, Ytr, model)
        dev_loss = evaluate(Xdev, Ydev, model)

        results.append((lr, train_loss, dev_loss))
        print(f"lr={lr:<4} train={train_loss:.4f} dev={dev_loss:.4f}")

    best = min(results, key=lambda x: x[2])
    print(f"\nBest learning rate: {best[0]}")

    print(f"Best dev loss: {best[2]:.4f}")

    return results




if __name__ == "__main__":
    compare_dimensions()
    learning_rate_sweep()
