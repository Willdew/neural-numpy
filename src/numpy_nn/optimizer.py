from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self, layer):
        """
        Updates the parameters of the given layer.
        """
        pass


class SGD(Optimizer):
    """
    Good ol' Stochastic Gradient Descent optimizer. Can also handle with momentum,
    which can lead to a lot better performance yay
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def update(self, layer):
        if hasattr(layer, "weights") and hasattr(layer, "weights_grad"):
            if self.momentum > 0:
                # With momentum
                if layer not in self.velocities:
                    self.velocities[layer] = {
                        "w": np.zeros_like(layer.weights),
                        "b": np.zeros_like(layer.bias),
                    }

                # Update velocities
                v = self.velocities[layer]
                v["w"] = (
                    self.momentum * v["w"] + self.learning_rate * layer.weights_grad
                )
                v["b"] = self.momentum * v["b"] + self.learning_rate * layer.bias_grad

                # Update weights using velocity
                layer.weights -= v["w"]
                layer.bias -= v["b"]
            else:
                # Without momentum
                layer.weights -= self.learning_rate * layer.weights_grad
                layer.bias -= self.learning_rate * layer.bias_grad
