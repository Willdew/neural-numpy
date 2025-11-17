from typing import List
from .layer import Layer
import numpy as np


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

    def backprop(self, loss_gradient: np.ndarray, learning_rate: float):
        grad = loss_gradient
        for layer in reversed(self.__layers):
            grad = layer.backward(grad, learning_rate)


# Builder method for constructing a new network, doesen't really do anything yet
# TODO: Make this acctually build a neural net, taking parameters such as number of layers, activation function and other good stuff
def BuildNetwork(self, num_layers: int, num_hidden_units: int) -> NeuralNetwork:
    return NeuralNetwork()
