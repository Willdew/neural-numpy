import os
from typing import List

import numpy as np
from rich import box, print
from rich.table import Table

import wandb
from data import DataLoader
from neural_numpy.builder import NetworkBuilder
from neural_numpy.confusion_matrix import confusion_matrix
from neural_numpy.loss import CategoricalCrossEntropy
from neural_numpy.optimizer import ADAM, SGD


def main():
    # 1. Setup WandB
    wandb.login(key=os.environ.get("WANDB_API_KEY"))
    run = wandb.init(
        project="azure_banditten",
        config={
            "epochs": 100,
            "learning_rate": 0.00909273148274152,
            "batch_size": 64,
            # Architecture
            "hidden_layers": 3,
            "hidden_units": 512,
            "weight_decay": 0.001,
            "activation": "ReLU",
            "optimizer": "sgd",
            "weight_initializer": "Xavier",
        },
    )
    config = wandb.config

    # New and improved data import
    X_train, y_train, X_test, y_test = DataLoader.load_cifar10(
        normalize=True, flatten=True, one_hot=True
    )
    print("[bold green]Data Loaded:[/bold green] CIFAR-10 Dataset")

    # subset_size = 2000
    # Split into training and validation data
    # X_train = X_train[:subset_size]
    # y_train = y_train[:subset_size]
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
    builder = NetworkBuilder()
    network = builder.build_from_wandb(
        input_size=input_dim, output_size=num_classes, config=config
    )

    if config.optimizer.lower() == "adam":
        optimizer = ADAM(
            learning_rate=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
    elif config.optimizer.lower() == "sgd":
        optimizer = SGD(
            learning_rate=config.learning_rate,
            weight_decay=getattr(config, "weight_decay", 0.0),
        )
    loss_fn = CategoricalCrossEntropy()

    network.train(
        X=X_train,
        y=y_train,
        X_val=X_val,
        y_val=y_val,
        loss_function=loss_fn,
        epochs=config.epochs,
        optimizer=optimizer,
        batch_size=config.batch_size,
    )

    y_true = np.argmax(y_test, axis=1)
    val_pred = network.forward(X_test)
    y_pred = np.argmax(val_pred, axis=1)
    val_acc_train = np.mean(np.argmax(val_pred, axis=1) == np.argmax(y_test, axis=1))
    print(f"[bold green]Test Accuracy: {val_acc_train:.1%}")

    cm = confusion_matrix(y_true, y_pred)
    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    num_classes = cm.shape[0]

    table = Table(title="Confusion Matrix", box=box.ROUNDED, show_lines=True)
    table.add_column("True\\Pred", style="dim", width=12)

    for i in range(num_classes):
        table.add_column(class_names[i], justify="right")

    for i in range(num_classes):
        row_data: List = []
        row_data.append(class_names[i])
        for j, val in enumerate(cm[i]):
            if j == i:
                row_data.append("[bold green]" + str(int(val)))
            else:
                row_data.append(str(int(val)))
        table.add_row(*row_data)
    print(table)
    run.finish()


if __name__ == "__main__":
    main()
