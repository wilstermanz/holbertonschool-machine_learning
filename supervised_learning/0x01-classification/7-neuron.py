#!/usr/bin/env python3
"""Task 7 - Neuron"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initializes a new Neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """W getter method"""
        return self.__W

    @property
    def b(self):
        """b getter method"""
        return self.__b

    @property
    def A(self):
        """A getter method"""
        return self.__A

    def forward_prop(self, X):
        """Calculates the forward propagation of the neuron"""
        n = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-n))
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)[1]
        return np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        A = self.forward_prop(X)
        prediction = A.round().astype(int)
        cost = self.cost(Y, A)
        return prediction, cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """Calculates one pass of gradient descent on the neuron"""
        m = np.shape(Y)[1]
        dz = A - Y
        dW = np.matmul(dz, X.T) / m
        db = np.sum(dz) / m
        self.__W -= alpha * dW
        self.__b -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neuron"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
            graph_data = {'x': [], 'y': []}
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            self.gradient_descent(X, Y, A, alpha)
            if verbose is True and i % step == 0:
                print("Cost after {} iterations: {}".format(i, self.cost(Y, A)))
            if graph is True and i % step == 0:
                graph_data['x'].append(i)
                graph_data['y'].append(self.cost(Y, A))          
        if graph is True:
            plt.plot(graph_data['x'], graph_data['y'], c='blue')
            plt.title('Training Cost')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.show()
        return self.evaluate(X, Y)
