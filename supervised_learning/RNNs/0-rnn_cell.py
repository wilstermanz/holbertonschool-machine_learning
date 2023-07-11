#!/usr/bin/env python3
"""Task 0"""
import numpy as np


class RNNCell:
    """represents a cell of a simple RNN"""

    def __init__(self, i, h, o):
        """
        Inputs:
            i: the dimensionality of the data
            h: the dimensionality of the hidden state
            o: the dimensionality of the outputs

        Creates the public instance attributes Wh, Wy, bh, by that represent
        the weights and biases of the cell
            Wh and bh are for the concatenated hidden state and input data
            Wy and by are for the output
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Inputs:
            x_t: a numpy.ndarray of shape (m, i) that contains the data input
            for the cell
                m is the batch size for the data
            h_prev: a numpy.ndarray of shape (m, h) containing the previous
            hidden state

        The output of the cell should use a softmax activation function

        Returns:
            h_next: the next hidden state
            y: the output of the cell
        """
        def softmax(x):
            max = np.max(x, axis=1, keepdims=True)
            return np.exp(x - max) / np.sum(
                np.exp(x - max), axis=1, keepdims=True)

        Whh = self.Wh[:h_prev.shape[1], :]
        Wxh = self.Wh[h_prev.shape[1]:, :]
        h_t = np.tanh((h_prev @ Whh) + (x_t @ Wxh) + self.bh)
        y_t = (h_t @ self.Wy) + self.by

        return h_t, softmax(y_t)
