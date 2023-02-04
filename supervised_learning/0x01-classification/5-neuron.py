#!/usr/bin/env python3
"""Task 3 - Neuron"""
import numpy as np


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
