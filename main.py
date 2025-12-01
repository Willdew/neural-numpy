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

    # 2. Get Data

    #Get CIFAR dataset
    X_train_cifar, y_train_cifar, X_test_cifar, y_test_cifar = DataLoader.load_cifar10(
        normalize=True,
        flatten=True,
        one_hot=True
    )
    print(f"[bold green]Data Loaded:[/bold green] CIFAR-10 Dataset")
    # For speed, take a subset
    subset_size_cifar = 2000
    X_cifar = X_train_cifar[:subset_size_cifar]
    y_cifar = y_train_cifar[:subset_size_cifar]
    print(f"[bold green]Using Subset:[/bold green] {subset_size_cifar} samples for training")
    # Make a validation set (20%)
    val_split = 0.2
    split_idx_cifar = int(X_cifar.shape[0] * (1 - val_split))
    X_val_cifar = X_cifar[split_idx_cifar:]
    y_val_cifar = y_cifar[split_idx_cifar:]
    X_cifar = X_cifar[:split_idx_cifar]
    y_cifar = y_cifar[:split_idx_cifar]
    print(f"[bold green]Training Set:[/bold green] {X_cifar.shape[0]} samples")
    print(f"[bold green]Validation Set:[/bold green] {X_val_cifar.shape[0]} samples")
    #The data loads!

    # Also get the data for MNIST
    #normalized, flattened, one-hot encoded data
    X_train_mnist, y_train_mnist, X_test_mnist, y_test_mnist = DataLoader.load_mnist(
    normalize=True,
    flatten=True,
    one_hot=True
    )
    # X_train: (60000, 784), y_train: (60000, 10)
    # X_test: (10000, 784), y_test: (10000, 10)
    print(f"[bold green]Data Loaded:[/bold green] MNIST Dataset")
    # For speed, take a subset
    subset_size_mnist = 2000
    X_mnist = X_train_mnist[:subset_size_mnist]
    y_mnist = y_train_mnist[:subset_size_mnist]
    print(f"[bold green]Using Subset:[/bold green] {subset_size_mnist} samples for training")
    # Make a validation set (20%)
    val_split = 0.2
    split_idx_mnist = int(X_mnist.shape[0] * (1 - val_split))
    X_val_mnist = X_mnist[split_idx_mnist:]
    y_val_mnist = y_mnist[split_idx_mnist:]
    X_mnist = X_mnist[:split_idx_mnist]
    y_mnist = y_mnist[:split_idx_mnist]
    print(f"[bold green]Training Set:[/bold green] {X_mnist.shape[0]} samples")
    print(f"[bold green]Validation Set:[/bold green] {X_val_mnist.shape[0]} samples")
    #The data loads!

    #just testing that data loads
    exit() 

    builder = NetworkBuilder()
    network = builder.build_from_wandb(input_size=2, output_size=2, config=wandb.config)

    # 4. Setup Training Components
    optimizer = SGD(learning_rate=config.learning_rate, momentum=config.momentum)
    loss_fn = CategoricalCrossEntropy()

    # 5. Train
    print("[bold blue]Starting Training...[/bold blue]")
    # Split data (example)
    X_train_cifar, X_val_cifar = X_cifar[:800], X_cifar[800:]
    y_train_cifar, y_val_cifar = y_cifar[:800], y_cifar[800:]

    network.train(
        X=X_train_cifar,
        y=y_train_cifar,
        X_val=X_val_cifar,  # Pass validation data here
        y_val=y_val_cifar,  # Pass validation labels here
        loss_function=loss_fn,
        epochs=config.epochs,
        optimizer=optimizer,
    )

    run.finish()


if __name__ == "__main__":
    main()
