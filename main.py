import os

import numpy as np
from rich import print

import wandb
from data import DataLoader
from neural_numpy.builder import NetworkBuilder
from neural_numpy.confusion_matrix import confusion_matrix
from neural_numpy.loss import CategoricalCrossEntropy
from neural_numpy.optimizer import ADAM


def main():
    # 1. Setup WandB
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run = wandb.init(
        project="skraldespanden",
        config={
            "epochs": 4,
            "learning_rate": 0.001,
            "batch_size": 128,
            # Architecture
            "hidden_layers": 3,
            "hidden_units": 512,
            "weight_decay": 1e-4,
            "activation": "ReLU",
            "output_activation": "Softmax",
            "weight_initializer": "He",
            "bias_initializer": "He",
        },
    )
    config = wandb.config

    # New and improved data import
    X_train, y_train, X_test, y_test = DataLoader.load_cifar10(
        normalize=True, flatten=True, one_hot=True
    )
    print("[bold green]Data Loaded:[/bold green] CIFAR-10 Dataset")

    # Split into training and validation data
    val_split = 0.2
    split_idx = int(X_train.shape[0] * (1 - val_split))

    # Validation data
    X_val = X_train[split_idx:]
    y_val = y_train[split_idx:]

    # Training data
    X = X_train[:split_idx]
    y = y_train[:split_idx]

    print(f"[bold green]Training Set:[/bold green] {X.shape[0]} samples")
    print(f"[bold green]Validation Set:[/bold green] {X_val.shape[0]} samples")

    # Automatically set dimensions based on data
    input_dim = X.shape[1]
    num_classes = y.shape[1]

    # Build network
    builder = NetworkBuilder()
    network = builder.build_from_wandb(
        input_size=input_dim, output_size=num_classes, config=wandb.config
    )

    # Setup setup optimizer and loss function
    optimizer = ADAM(learning_rate=config.learning_rate)
    loss_fn = CategoricalCrossEntropy()

    # Train network yay
    print("[bold blue]Starting Training...[/bold blue]")
    network.train(
        X=X,
        y=y,
        X_val=X_val,
        y_val=y_val,
        loss_function=loss_fn,
        epochs=config.epochs,
        optimizer=optimizer,
    )
    y_true = np.argmax(y_val, axis=1)
    y_pred = np.argmax(network.forward(X_val), axis=1)
    cm = confusion_matrix(y_true, y_pred)
    print(
        "\n[bold blue]Confusion Matrix (true (rows) / predicted (columns)):[/bold blue]"
    )
    print(cm)
    run.finish()


if __name__ == "__main__":
    main()
