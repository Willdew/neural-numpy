import numpy as np
import wandb
import os
from rich import print
from data import DataLoader, one_hot_encode
from neural_numpy.builder import NetworkBuilder, ActivationType, InitializerType
from neural_numpy.loss import MSE, CategoricalCrossEntropy
from neural_numpy.optimizer import SGD


def generate_circles_data(n_samples=500):
    """
    Generates a dataset of concentric circles.
    Class 0: Inner circle (radius < 0.5)
    Class 1: Outer ring (0.5 < radius < 1.0)
    """
    np.random.seed(42)
    X = (np.random.rand(n_samples, 2) - 0.5) * 2.5  # Range [-1.25, 1.25]

    # Calculate radius squared: x^2 + y^2
    radius_sq = np.sum(X**2, axis=1)

    # Create labels based on radius
    # Class 0: Inside circle of radius 0.6
    # Class 1: Outside that circle
    y_indices = (radius_sq > 0.6**2).astype(int)

    # Convert to one-hot: (N, 2)
    y = np.zeros((n_samples, 2))
    y[np.arange(n_samples), y_indices] = 1

    return X, y


def main():
    # 1. Setup WandB
    wandb.login(key=os.environ.get("WANDB_API_KEY"))

    run = wandb.init(
        project="skraldespanden",
        config={
            "epochs": 2000,
            "learning_rate": 0.05,
            "batch_size": 32,
            "momentum": 0.9,
            # Architecture
            "hidden_layers": 2,
            "hidden_units": 16,
            "activation": "Tanh",
            "output_activation": "Softmax",
            "weight_initializer": "Xavier",
        },
    )
    config = wandb.config

    # 2. Get Data (Circles)
    #X, y = generate_circles_data(n_samples=1000)
    #print(f"[bold green]Data Generated:[/bold green] Concentric Circles (1000 samples)")
    #New and improved data import
    X_train, y_train, X_test, y_test = DataLoader.load_cifar10(
        normalize=True,
        flatten=True,
        one_hot=True
    )
    print(f"[bold green]Data Loaded:[/bold green] CIFAR-10 Dataset")
    # For speed, take a subset
    subset_size = 2000
    X = X_train[:subset_size]
    y = y_train[:subset_size]
    print(f"[bold green]Using Subset:[/bold green] {subset_size} samples for training")
    # Make a validation set (20%)
    val_split = 0.2
    split_idx = int(X.shape[0] * (1 - val_split))
    X_val = X[split_idx:]
    y_val = y[split_idx:]
    X = X[:split_idx]
    y = y[:split_idx]
    print(f"[bold green]Training Set:[/bold green] {X.shape[0]} samples")
    print(f"[bold green]Validation Set:[/bold green] {X_val.shape[0]} samples")
    exit()
    #The data loads!

    builder = NetworkBuilder()
    network = builder.build_from_wandb(input_size=2, output_size=2, config=wandb.config)

    # 4. Setup Training Components
    optimizer = SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    loss_fn = CategoricalCrossEntropy()

    # 5. Train
    print("[bold blue]Starting Training...[/bold blue]")
    # Split data (example)
    X_train, X_val = X[:800], X[800:]
    y_train, y_val = y[:800], y[800:]

    network.train(
        X=X_train,
        y=y_train,
        X_val=X_val,  # Pass validation data here
        y_val=y_val,  # Pass validation labels here
        loss_function=loss_fn,
        epochs=config.epochs,
        optimizer=optimizer,
    )

    run.finish()


if __name__ == "__main__":
    main()
