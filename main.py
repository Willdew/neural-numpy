from fileinput import filename
import numpy as np
import pickle
from numpy_nn.network import NeuralNetwork
from numpy_nn.layer import Dense
from numpy_nn.activation import Softmax, Tanh
from numpy_nn.loss import MSE
from numpy_nn.initializers import Xavier, Zeros

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
    #np.random.seed(42)
    
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

    #Let's take a fraction of the entire dataset to speed up training
    train_size = 10000
    test_size = 2000

    X_train_small = X_train[:train_size].reshape(train_size, -1) / 255.0
    y_train_small = y_train[:train_size]
    y_train_small = one_hot_encode(y_train_small, num_classes=10) #convert to one hot encoding


    #Train a simple network
   
    #Multi-class classification with 10 classes
    network = NeuralNetwork()
    
    #Input layer - 3072 inputs (32*32*3)
    network.add_layer(Dense(input_size=3072, output_size=1024, activation=Tanh()))

    #Hidden layers
    network.add_layer(Dense(1024, 512, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 1
    network.add_layer(Dense(512, 256, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 2
    network.add_layer(Dense(256, 128, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 3
    network.add_layer(Dense(128, 64, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 4
    network.add_layer(Dense(64, 32, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 5
    network.add_layer(Dense(32, 16, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 6
    #network.add_layer(Dense(1024, 64, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 7
    #network.add_layer(Dense(1024, 64, Tanh(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Hidden Layer 8
    #Add another hidden layer

    #Output layer
    network.add_layer(Dense(16, 10, Tanh(), activation=Softmax(), weight_initializer=Xavier(), bias_initializer=Zeros())) #Output Layer



    network.train(X_train_small, y_train_small, MSE(), epochs=100, learning_rate=0.01, printProgressRate=5)

    #Test accuracy on validation set
    #Use the samples from the back of the full training set
    X_validation = X_train[-test_size:].reshape(test_size, -1) / 255.0
    y_validation = y_train[-test_size:]
    predictions = network.forward(X_validation)
    predicted_classes = np.argmax(predictions, axis=1)
    accuracy = np.mean(predicted_classes == y_validation)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")



def one_hot_encode(labels, num_classes=10):
    """Convert integer labels to one-hot encoded format"""
    one_hot = np.zeros((labels.shape[0], num_classes))
    one_hot[np.arange(labels.shape[0]), labels] = 1
    return one_hot


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
