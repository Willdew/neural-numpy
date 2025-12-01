from typing import List

from neural_numpy.optimizer import Optimizer
from .layer import Layer
import numpy as np
import wandb
from rich.progress import (
    Progress,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
)


class NeuralNetwork:
    def __init__(self):
        self.__layers: List[Layer] = []

    # Adds layers what a surprise
    def add_layer(self, layer: Layer):
        self.__layers.append(layer)

    # Gets the layers
    def get_layers(self):
        return self.__layers

    # These are not really finished, more as a reference
    # forward does not have side effects at the moment, backprop does
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        output = input_data
        for layer in self.__layers:
            output = layer.forward(output)
        return output

    def backprop(self, loss_gradient: np.ndarray):
        grad = loss_gradient
        for layer in reversed(self.__layers):
            grad = layer.backward(grad)

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        loss_function,
        epochs: int,
        optimizer: Optimizer,
        batch_size: int = 32,
    ):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]Training...", total=epochs)

            n_samples = X.shape[0]

            for epoch in range(epochs):
                # Shuffle data at the start of each epoch
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                X_shuffled = X[indices]
                y_shuffled = y[indices]

                epoch_loss = 0
                epoch_acc = 0
                num_batches = 0

                # Mini-batch loop
                for start_idx in range(0, n_samples, batch_size):
                    end_idx = min(start_idx + batch_size, n_samples)
                    X_batch = X_shuffled[start_idx:end_idx]
                    y_batch = y_shuffled[start_idx:end_idx]

                    # 1. Forward
                    predictions = self.forward(X_batch)

                    # 2. Loss
                    loss = loss_function.forward(predictions, y_batch)
                    loss_gradient = loss_function.backward(predictions, y_batch)

                    # Accumulate metrics
                    epoch_loss += loss
                    batch_acc = np.mean(
                        np.argmax(predictions, axis=1) == np.argmax(y_batch, axis=1)
                    )
                    epoch_acc += batch_acc
                    num_batches += 1

                    # 3. Zero Gradients
                    for layer in self.__layers:
                        optimizer.zero_grad(layer)

                    # 4. Backward
                    self.backprop(loss_gradient)

                    # 5. Optimizer Step
                    for layer in self.__layers:
                        optimizer.step(layer)

                # Average metrics over all batches
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_acc / num_batches

                # Update WandB
                wandb.log({"loss": float(avg_loss), "acc": float(avg_acc)})

                progress.update(
                    task_id,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{epochs} [magenta]Loss: {avg_loss:.4f} [green]Acc: {avg_acc:.1%}",
                )
