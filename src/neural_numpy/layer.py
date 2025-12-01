# src/neural_numpy/layer.py
from abc import ABC, abstractmethod
import numpy as np
from .initializers import Initializer


class Parameter:
    """
    Wraps a learnable parameter (tensor) and its gradient.
    """

    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        """Reset gradients to zero."""
        self.grad.fill(0)


class Layer(ABC):
    """
    Abstract base class for a neural network layer.
    This isn't used directly, but used as a base for other layers
    """

    def __init__(self):
        self.input = None
        self.output = None
        self.input_gradient = None

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

    def get_parameters(self):
        """
        Returns a list of dictionaries for learnable parameters.
        Format: [{'param': p, 'grad': g, 'name': 'w'}, ...]
        """
        return []


class Dense(Layer):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        weight_initializer: Initializer,
        bias_initializer: Initializer,
    ):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()

        # Initialize weights and biases
        self.weights = Parameter(
            weight_initializer.initialize((input_size, output_size))
        )

        self.bias = Parameter(bias_initializer.initialize((1, output_size)))

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        return np.dot(self.input, self.weights.data) + self.bias.data

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        self.weights.grad += np.dot(self.input.T, output_gradient)
        if self.bias:
            self.bias.grad += np.sum(output_gradient, axis=0, keepdims=True)

        return np.dot(output_gradient, self.weights.data.T)

    def get_parameters(self):
        params = [self.weights]
        if self.bias:
            params.append(self.bias)
        return params
