# src/numpy_nn/layer.py
from abc import ABC, abstractmethod
import numpy as np


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
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Computes the backward pass (truly astonishing).
        Returns input_gradient (dL/dInput).
        """
        pass


class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, activation: Layer):
        super().__init__()

        # initialize the weights
        # TODO: We should probably pass the initializer to the constructor
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.bias = np.zeros((1, output_size))
        self.activation = activation

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.z = np.dot(self.input, self.weights) + self.bias
        return self.activation.forward(self.z)

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # use activation to find output gradient
        output_gradient = self.activation.backward(output_gradient, learning_rate)

        # 2. Calculate gradients for weights and biases
        weights_gradient = np.dot(self.input.T, output_gradient)
        bias_gradient = np.sum(output_gradient, axis=0, keepdims=True)

        # 3. Calculate gradient for the input (to pass to the previous layer)
        input_gradient = np.dot(output_gradient, self.weights.T)

        # 4. Update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * bias_gradient

        return input_gradient
