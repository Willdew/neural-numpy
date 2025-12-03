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
    Good ol' Stochastic Gradient Descent optimizer.
    """

    def __init__(self, learning_rate: float = 0.01):
        super().__init__(learning_rate)
        self.velocities = {}

    def step(self, layer):
        for param in layer.get_parameters():
            if param.grad is None:
                continue

            param.data -= self.learning_rate * param.grad


class ADAM(Optimizer):
    """
    Adam optimizer, our favorite optimizer for deep learning :))
    It adapts the learning rate for each parameter based on the first and second moments of the gradients.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self, layer):
        self.t += 1
        for param in layer.get_parameters():
            if param.grad is None:
                continue

            if self.weight_decay > 0:
                param.grad += self.weight_decay * param.data

            p_id = id(param)

            if p_id not in self.m:
                self.m[p_id] = np.zeros_like(param.data)
                self.v[p_id] = np.zeros_like(param.data)

            self.m[p_id] = self.beta1 * self.m[p_id] + (1 - self.beta1) * param.grad
            self.v[p_id] = self.beta2 * self.v[p_id] + (1 - self.beta2) * (
                param.grad**2
            )

            m_hat = self.m[p_id] / (1 - self.beta1**self.t)
            v_hat = self.v[p_id] / (1 - self.beta2**self.t)

            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
