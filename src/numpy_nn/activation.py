import numpy as np
from .layer import Layer


class Activation(Layer):
    """
    Base class for activation functions.
    Applies a function to inputs element-wise, is implemented as a layer
    """

    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        self.input = input_data
        self.output = self.activation(self.input)
        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return np.multiply(output_gradient, self.activation_prime(self.input))


# Activation Functions - See Lecture 1 slide 38


# Sigmoid Activation Function
class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            # Clip to prevent overflow
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

        def sigmoid_prime(x):
            s = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)


# Arc-tangent Activation Function
class ArcTan(Activation):
    def __init__(self):
        def arctan(x):
            return np.arctan(x)

        def arctan_prime(x):
            return 1.0 / (1.0 + np.square(x))

        super().__init__(arctan, arctan_prime)


# Hyperbolic Tangent Activation Function
class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)

        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)


# Rectified Linear Unit Activation Function
class ReLU(Activation):
    def __init__(self):
        def relu(x):
            return np.maximum(0, x)

        def relu_prime(x):
            return (x > 0).astype(float)

        super().__init__(relu, relu_prime)


# For output layer


# Softmax is used for Classification
class Softmax(Layer):
    """
    Softmax activation for multi-class classification.
    Converts logits to probability distribution.

    Output: probabilities that sum to 1.0
    Use with CategoricalCrossEntropy loss.
    """

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute softmax: exp(x_i) / sum(exp(x_j))
        Uses numerical stability trick: subtract max before exp
        """
        self.input = input_data

        # Subtract max for numerical stability (prevents overflow)
        exp_values = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))

        # Normalize to get probabilities
        self.output = exp_values / np.sum(exp_values, axis=-1, keepdims=True)

        return self.output

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """
        Compute gradient of softmax.

        Note: When used with CategoricalCrossEntropy, the combined gradient
        simplifies to (y_pred - y_true). This is the full Jacobian approach.
        """
        # Number of samples
        n = output_gradient.shape[0]

        # For each sample, compute Jacobian and multiply with output gradient
        input_gradient = np.empty_like(output_gradient)

        for i in range(n):
            # Reshape for matrix operations
            output = self.output[i].reshape(-1, 1)

            # Jacobian matrix: S_ij = S_i * (δ_ij - S_j)
            jacobian = np.diagflat(output) - np.dot(output, output.T)

            # Multiply Jacobian with gradient
            input_gradient[i] = np.dot(jacobian, output_gradient[i])

        return input_gradient
