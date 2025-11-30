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
        params = layer.get_parameters()
        if not params:
            return

        # Initialize velocity cache for this layer if needed
        if self.momentum > 0 and layer not in self.velocities:
            self.velocities[layer] = [np.zeros_like(p["param"]) for p in params]

        for i, p_dict in enumerate(params):
            param = p_dict["param"]
            grad = p_dict["grad"]

            if grad is None:
                continue

            if self.momentum > 0:
                self.velocities[layer][i] = (
                    self.momentum * self.velocities[layer][i]
                    + self.learning_rate * grad
                )

                # Update parameter: w = w - v
                param -= self.velocities[layer][i]
            else:
                # Standard SGD: w = w - lr * grad
                param -= self.learning_rate * grad
