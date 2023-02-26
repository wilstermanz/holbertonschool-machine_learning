#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent
    with L2 regularization:

        * Y is a one-hot numpy.ndarray of shape (classes, m) that contains the
          correct labels for the data
            * classes is the number of classes
            * m is the number of data points
        * weights is a dictionary of the weights and biases of the neural
          network
        * cache is a dictionary of the outputs of each layer of the neural
          network
        * alpha is the learning rate
        * lambtha is the L2 regularization parameter
        * L is the number of layers of the network
        * The neural network uses tanh activations on each layer except the
          last, which uses a softmax activation
        * The weights and biases of the network should be updated in place
    """
    m = np.shape(Y)[1]
    input_layer = 1
    output_layer = L
    for layer in range(output_layer, 0, -1):
        a = cache["A{}".format(layer)]
        aprev = cache["A{}".format(layer - 1)]
        W = weights["W{}".format(layer)]
        b = weights["b{}".format(layer)]
        l2 = (lambtha / m) + W
        if layer == output_layer:
            dz = a - Y
        if layer < output_layer:
            dz = da * (1 - np.square(a))
        dW = np.matmul(dz, aprev.T) / m + l2
        db = np.sum(dz, axis=1, keepdims=True) / m
        da = np.matmul(W.T, dz)
        W -= alpha * dW
        b -= alpha * db
