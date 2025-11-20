from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """
    Abstract base class for loss functions.
    """

    @abstractmethod
    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
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

    def forward(self, y_pred: np.ndarray, y_true: np.ndarray) -> float:
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

