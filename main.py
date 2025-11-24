from fileinput import filename
import numpy as np
import pickle
from numpy_nn.network import NeuralNetwork
from numpy_nn.layer import Dense
from numpy_nn.activation import Tanh
from numpy_nn.loss import MSE

import urllib.request
import tarfile
import os

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
    #test_xor()
    #test_simple_regression()
    downloadDataset()
    X_train = []
    y_train = []
    
    for i in range(1, 6):
        imgs, labels = load_batch(f"./cifar-10-batches-py/data_batch_{i}")
        X_train.append(imgs)
        y_train.append(labels)
    
    X_train = np.concatenate(X_train, axis=0)
    y_train = np.concatenate(y_train, axis=0)
    
    X_test, y_test = load_batch("./cifar-10-batches-py/test_batch")
    
    print(X_train.shape)   # (50000, 32, 32, 3)
    print(y_train.shape)   # (50000,)
    
    print(X_test.shape)    # (10000, 32, 32, 3)
    print(y_test.shape)    # (10000,)


def downloadDataset():
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"

    # Download file
    if not os.path.exists(filename):
        print("Downloading CIFAR-10...")
        urllib.request.urlretrieve(url, filename)

    # Extract it
    if not os.path.exists("./cifar-10-batches-py"):
        print("Extracting...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()

def load_batch(batch_path):
    with open(batch_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    # Convert keys from bytes to strings
    batch = {k.decode(): v for k, v in batch.items()}
    
    data = batch["data"]               # shape: (10000, 3072)
    labels = batch["labels"]           # list of 10000 integers
    filenames = batch["filenames"]     # list of names

    # Reshape to images: (10000, 32, 32, 3)
    images = data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
    
    return images, np.array(labels)



if __name__ == "__main__":
    main()
