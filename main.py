import numpy as np
from numpy_nn.network import NeuralNetwork
from numpy_nn.layer import Dense
from numpy_nn.activation import Activation, Tanh


def main():
    network = NeuralNetwork()
    network.add_layer(Dense(2, 2, Tanh()))
    network.add_layer(Dense(2, 2, Tanh()))
    print(network.forward(np.array([3, 3])))

    #TODO: implement training and prediction example
    #network.train(X_train, y_train, epochs=1000, learning_rate=0.01)
    #prediction = network.predict(X_test)


if __name__ == "__main__":
    main()

