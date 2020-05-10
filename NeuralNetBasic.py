import numpy as np


def sigmoid(z):
    return 1/(1+np.exp(-z))


def sigmoid_derivative(z):
    enz = np.exp(-z)
    return enz/(enz + 1)**2


class NeuralNetwork:
    # copied directly from https://towardsdatascience.com/how-to-build-your-own-neural-network-from-scratch-in-python-68998a08e4f6
    # then edited some
    def __init__(self, x, y):
        self.input      = x
        self.weights1   = np.random.rand(self.input.shape[1],4)
        self.biases1    = np.random.rand(self.input.shape[1],4)
        self.weights2   = np.random.rand(4,1)
        self.biases2    = np.random.rand(4,1)
        self.y          = y
        self.output     = np.zeros(self.y.shape)

    def feedforward(self):
        self.layer1 = sigmoid(np.dot(self.input, self.weights1) + self.biases1)
        self.output = sigmoid(np.dot(self.layer1, self.weights2) + self.biases2)

    def backprop(self):
        # application of the chain rule to find derivative of the loss function with respect to weights2 and weights1
        d_weights2 = np.dot(self.layer1.T, (2*(self.y - self.output) * sigmoid_derivative(self.output)))
        d_weights1 = np.dot(self.input.T,  (np.dot(2*(self.y - self.output) * sigmoid_derivative(self.output), self.weights2.T) * sigmoid_derivative(self.layer1)))

        # update the weights with the derivative (slope) of the loss function
        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def print(self):
        print("weights1:")
        print(self.weights1)
        print("biases1:")
        print(self.biases1)
        print("weights2:")
        print(self.weights2)
        print("biases2")
        print(self.biases2)
        print("----")


if __name__ == "__main__":
    x = np.array([[0,0,1], [0,1,1], [1,0,1], [1,1,1]]).T  # samples are column vectors
    y = np.array([[0], [1], [1], [0]])
    net = NeuralNetwork(x, y)
    while True:
        net.feedforward()
        net.backprop()
        net.print()
