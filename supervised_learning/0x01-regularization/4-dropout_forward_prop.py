#!/usr/bin/env python3
"""Task 4"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    conducts forward propagation using Dropout:

        * X is a numpy.ndarray of shape (nx, m) containing the input data for
          the network
            * nx is the number of input features
            * m is the number of data points
        * weights is a dictionary of the weights and biases of the neural
          network
        * L the number of layers in the network
        * keep_prob is the probability that a node will be kept
        * All layers except the last should use the tanh activation function
        * The last layer should use the softmax activation function
        * Returns: a dictionary containing the outputs of each layer and the
          dropout mask used on each layer (see example for format)
    """
    outputs = {}
    outputs['A0'] = X

    for layer in range(1, L + 1):
        W = weights['W' + str(layer)]
        A = outputs['A' + str(layer - 1)]
        b = weights['b' + str(layer)]
        z = np.matmul(W, A) + b

        e = np.exp
        if layer == L:
            outputs['A' + str(layer)] = e(z) / np.sum(e(z), axis=0)
        else:
            A = (e(z) - e(-z)) / (e(z) + e(-z))
            d = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            outputs['D' + str(layer)] = d.astype(int)
            A *= d
            A /= keep_prob
            outputs['A' + str(layer)] = A

    return outputs
