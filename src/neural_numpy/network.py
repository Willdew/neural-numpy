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
    ):
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task_id = progress.add_task("[cyan]Training...", total=epochs)

        for epoch in range(epochs):
            # 1. Forward
            predictions = self.forward(X)

            # 2. Loss
            loss = loss_function.forward(predictions, y)
            loss_gradient = loss_function.backward(predictions, y)

            # 3. Zero Gradients
            for layer in self.__layers:
                optimizer.zero_grad(layer)

            # 4. Backward
            self.backprop(loss_gradient)

            # 5. Optimizer Step
            for layer in self.__layers:
                optimizer.step(layer)

                # 5. Metrics & Logging
                acc = np.mean(np.argmax(predictions, axis=1) == np.argmax(y, axis=1))

                # Update WandB
                wandb.log({"loss": float(loss), "acc": float(acc)})

                progress.update(
                    task_id,
                    advance=1,
                    description=f"[cyan]Epoch {epoch + 1}/{epochs} [magenta]Loss: {loss:.4f} [green]Acc: {acc:.1%}",
                )


# Builder method for constructing a new network, doesen't really do anything yet
# TODO: Make this acctually build a neural net, taking parameters such as number of layers, activation function and other good stuff
def BuildNetwork(self, num_layers: int, num_hidden_units: int) -> NeuralNetwork:
    return NeuralNetwork()
