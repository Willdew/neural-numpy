import numpy as np
from numpy_nn.network import NeuralNetwork
from numpy_nn.layer import Dense
from numpy_nn.activation import Tanh
from numpy_nn.loss import MSE


def test_xor():
    """Test the network on the XOR problem - a classic non-linear problem"""
    print("=" * 50)
    print("Testing XOR Problem")
    print("=" * 50)
    
    # XOR dataset
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    
    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    
    # Build network: 2 inputs -> 3 hidden -> 1 output
    network = NeuralNetwork()
    network.add_layer(Dense(2, 3, Tanh()))
    network.add_layer(Dense(3, 1, Tanh()))
    
    # Train
    print("\nTraining...")
    loss_fn = MSE()
    network.train(X, y, loss_fn, epochs=1000, learning_rate=0.1)
    
    # Test predictions
    print("\n" + "=" * 50)
    print("Final Predictions:")
    print("=" * 50)
    for i, (input_data, target) in enumerate(zip(X, y)):
        prediction = network.forward(input_data.reshape(1, -1))
        print(f"Input: {input_data} | Target: {target[0]} | Prediction: {prediction[0, 0]:.4f}")


def test_simple_regression():
    """Test on simple linear regression: y = 2x + 1"""
    print("\n\n" + "=" * 50)
    print("Testing Simple Regression: y = 2x + 1")
    print("=" * 50)
    
    # Generate data
    X = np.array([[0], [1], [2], [3], [4]])
    y = np.array([[1], [3], [5], [7], [9]])  # y = 2x + 1
    
    # Build network: 1 input -> 2 hidden -> 1 output
    network = NeuralNetwork()
    network.add_layer(Dense(1, 2, Tanh()))
    network.add_layer(Dense(2, 1, Tanh()))
    
    # Train
    print("\nTraining...")
    loss_fn = MSE()
    network.train(X, y, loss_fn, epochs=1000, learning_rate=0.01)
    
    # Test predictions
    print("\n" + "=" * 50)
    print("Final Predictions:")
    print("=" * 50)
    for i, (input_data, target) in enumerate(zip(X, y)):
        prediction = network.forward(input_data.reshape(1, -1))
        print(f"Input: {input_data[0]} | Target: {target[0]} | Prediction: {prediction[0, 0]:.4f}")


def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Run tests
    test_xor()
    test_simple_regression()


if __name__ == "__main__":
    main()
