from abc import ABC, abstractmethod
import numpy as np


class Optimizer(ABC):
    """
    Abstract base class for optimizers.
    """

    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    @abstractmethod
    def step(self, layer):
        """
        Updates the parameters of the given layer.
        """
        pass

    def zero_grad(self, layer):
        """
        Clears gradients for the layer.
        Should be called before backward().
        """
        for param in layer.get_parameters():
            param.zero_grad()


class SGD(Optimizer):
    """
    Good ol' Stochastic Gradient Descent optimizer. Can also handle with momentum,
    which can lead to a lot better performance yay
    """

    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        super().__init__(learning_rate)
        self.momentum = momentum
        self.velocities = {}

    def step(self, layer):
        for param in layer.get_parameters():
            if param.grad is None:
                continue

            p_id = id(param)

            if self.momentum > 0:
                if p_id not in self.velocities:
                    self.velocities[p_id] = np.zeros_like(param.data)

                self.velocities[p_id] = (
                    self.momentum * self.velocities[p_id]
                    + self.learning_rate * param.grad
                )

                param.data -= self.velocities[p_id]
            else:
                param.data -= self.learning_rate * param.grad
