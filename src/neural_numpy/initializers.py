from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    """
    Abstract base class for weight initialization strategies.
    """

    @abstractmethod
    def initialize(self, shape: tuple) -> np.ndarray:
        """
        Initialize weights with given shape.

        Args:
            shape: Tuple of (input_size, output_size)

        Returns:
            Initialized weight matrix
        """
        pass


class RandomNormal(Initializer):
    """
    Samples from N(0, std²)
    """

    def __init__(self, std: float = 0.1):
        """
        Args:
            std: Standard deviation of the normal distribution
        """
        self.std = std

    def initialize(self, shape: tuple) -> np.ndarray:
        return np.random.randn(*shape) * self.std


class Xavier(Initializer):
    """
    Xavier/Glorot initialization.
    Best for: Tanh, Sigmoid activations

    Variance = 2 / (fan_in + fan_out)

    Keeps variance of activations roughly the same across layers,
    preventing vanishing/exploding gradients.
    """

    def __init__(self, uniform: bool = True):
        """
        Args:
            uniform: If True, use uniform distribution; else normal distribution
        """
        self.uniform = uniform

    def initialize(self, shape: tuple) -> np.ndarray:
        fan_in, fan_out = shape
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        if self.uniform:
            # Uniform distribution in [-limit, limit]
            return np.random.uniform(-limit, limit, size=shape)
        else:
            # Normal distribution with std = sqrt(2 / (fan_in + fan_out))
            std = np.sqrt(2.0 / (fan_in + fan_out))
            return np.random.randn(*shape) * std


class He(Initializer):
    """
    He initialization (Kaiming initialization).
    Best for: ReLU, Leaky ReLU activations

    Variance = 2 / fan_in

    Designed specifically for ReLU networks to maintain variance
    when half the units are deactivated.
    """

    def __init__(self, uniform: bool = False):
        """
        Args:
            uniform: If True, use uniform distribution; else normal distribution
        """
        self.uniform = uniform

    def initialize(self, shape: tuple) -> np.ndarray:
        fan_in = shape[0]

        if self.uniform:
            # Uniform distribution
            limit = np.sqrt(6.0 / fan_in)
            return np.random.uniform(-limit, limit, size=shape)
        else:
            # Normal distribution (more common)
            std = np.sqrt(2.0 / fan_in)
            return np.random.randn(*shape) * std


class Zeros(Initializer):
    """
    Initialize all weights to zero.
    """

    def initialize(self, shape: tuple) -> np.ndarray:
        return np.zeros(shape)


class Ones(Initializer):
    """
    Initialize all weights to one.

    Rarely used - mainly for testing or specific architectures.
    """

    def initialize(self, shape: tuple) -> np.ndarray:
        return np.ones(shape)


class Constant(Initializer):
    """
    Initialize all weights to a constant value.
    """

    def __init__(self, value: float = 0.0):
        """
        Args:
            value: Constant value to initialize with
        """
        self.value = value

    def initialize(self, shape: tuple) -> np.ndarray:
        return np.full(shape, self.value)
