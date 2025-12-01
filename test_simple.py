import numpy as np
from neural_numpy.network import NeuralNetwork
from neural_numpy.layer import Dense
from neural_numpy.activation import Tanh
from neural_numpy.loss import MSE

# Set seed
np.random.seed(42)

# Simple data
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

print("Data shapes:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")

# Build network
network = NeuralNetwork()
network.add_layer(Dense(2, 3, Tanh()))
network.add_layer(Dense(3, 1, Tanh()))

print("\nTesting forward pass...")
try:
    output = network.forward(X)
    print(f"Forward pass successful! Output shape: {output.shape}")
    print(f"Output:\n{output}")
except Exception as e:
    print(f"Error in forward pass: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting loss computation...")
try:
    loss_fn = MSE()
    loss = loss_fn.forward(output, y)
    print(f"Loss computed: {loss}")
    
    loss_grad = loss_fn.backward(output, y)
    print(f"Loss gradient shape: {loss_grad.shape}")
    print(f"Loss gradient:\n{loss_grad}")
except Exception as e:
    print(f"Error in loss: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting backward pass...")
try:
    network.backprop(loss_grad, 0.1)
    print("Backward pass successful!")
except Exception as e:
    print(f"Error in backward pass: {e}")
    import traceback
    traceback.print_exc()

print("\nTesting one training epoch...")
try:
    network.train(X, y, loss_fn, epochs=1, learning_rate=0.1)
    print("Training epoch successful!")
except Exception as e:
    print(f"Error in training: {e}")
    import traceback
    traceback.print_exc()
