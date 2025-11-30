from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Computes the loss value.

        Args:
            y_pred: Predicted values
            y_true: True/target values

        Returns:
            Scalar loss value
        """
        pass

    @abstractmethod
    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Computes the gradient of the loss with respect to predictions.

        Args:
            y_pred: Predicted values
            y_true: True/target values

        Returns:
            Gradient dL/dy_pred (same shape as y_pred)
        """
        pass


class MSE(Loss):
    """
    Mean Squared Error loss.
    Best for regression problems.

    L = (1/n) * Σ(y_pred - y_true)²
    """

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Compute MSE loss.
        """
        return np.mean((y_pred - y_true) ** 2)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient: dL/dy_pred = 2 * (y_pred - y_true) / n
        """
        n = y_pred.shape[0] if len(y_pred.shape) > 1 else 1
        return 2 * (y_pred - y_true) / n


class BinaryCrossEntropy(Loss):
    """
    Binary Cross-Entropy loss.
    Best for binary classification (0 or 1).

    L = -1/n * Σ(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))
    """

    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small value to prevent log(0)
        """
        self.epsilon = epsilon

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Compute binary cross-entropy loss.
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        loss = -np.mean(
            y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log(1 - y_pred_clipped)
        )
        return loss

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient: dL/dy_pred = -(y_true/y_pred - (1-y_true)/(1-y_pred)) / n
        """
        # Clip to prevent division by zero
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        n = y_pred.shape[0] if len(y_pred.shape) > 1 else 1
        gradient = -(y_true / y_pred_clipped - (1 - y_true) / (1 - y_pred_clipped)) / n

        return gradient


class CategoricalCrossEntropy(Loss):
    """
    Categorical Cross-Entropy loss.
    Best for multi-class classification with one-hot encoded labels.

    L = -1/n * Σ Σ(y_true * log(y_pred))
    """

    def __init__(self, epsilon: float = 1e-15):
        """
        Args:
            epsilon: Small value to prevent log(0)
        """
        self.epsilon = epsilon

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.floating:
        """
        Compute categorical cross-entropy loss.
        """
        # Clip predictions to prevent log(0)
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        # Sum over classes, mean over batch
        loss = -np.mean(np.sum(y_true * np.log(y_pred_clipped), axis=-1))
        return loss

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """
        Compute gradient: dL/dy_pred = -y_true / y_pred / n
        """
        # Clip to prevent division by zero
        y_pred_clipped = np.clip(y_pred, self.epsilon, 1 - self.epsilon)

        n = y_pred.shape[0]
        gradient = -y_true / y_pred_clipped / n

        return gradient
