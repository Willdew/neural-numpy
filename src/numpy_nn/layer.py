# src/numpy_nn/layer.py
from abc import ABC, abstractmethod
import numpy as np
from .initializers import Initializer


class Layer(ABC):
    """
    Abstract base class for a neural network layer.
    This isn't used directly, but used as a base for other layers
    """

    def __init__(self):
        self.input = None
        self.output = None

    @abstractmethod
    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Computes the forward pass (what a surprise).
        Stores input for backprop usage.
        """
        pass

    @abstractmethod
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Computes the backward pass (truly astonishing).
        Returns input_gradient (dL/dInput).
        """
        pass


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_initializer: Initializer,
        bias_initializer: Initializer,
    ):
        super().__init__()

        # Initialize weights and biases
        self.weights = weight_initializer.initialize((input_size, output_size))

        # Biases typically initialized to zero
        if bias_initializer is not None:
            self.bias = bias_initializer.initialize((1, output_size))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.dot(self.input, self.weights) + self.bias

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.weights_gradient = np.dot(self.input.T, output_gradient)
        self.bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        return np.dot(output_gradient, self.weights.T)
