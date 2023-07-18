#!/usr/bin/env python3
"""Task 7"""
import numpy as np


class BidirectionalCell:
    """represents a bidirectional cell of an RNN"""

    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden states
        o is the dimensionality of the outputs
        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by that
        represent the weights and biases of the cell
            Whf and bhf are for the hidden states in the forward direction
            Whb and bhb are for the hidden states in the backward direction
            Wy and by are for the outputs
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Whf = np.random.randn(i+h, h)
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.randn(i+h, h)
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.randn(h*2, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        calculates the hidden state in the forward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        Returns: h_next, the next hidden state
        """
        h_prev_x_t = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh((h_prev_x_t @ self.Whf) + self.bhf)

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batch size for the data
        h_next is a numpy.ndarray of shape (m, h) containing the next hidden
        state
        Returns: h_prev, the previous hidden state
        """
        h_next_x_t = np.concatenate((h_next, x_t), axis=1)
        return np.tanh((h_next_x_t @ self.Whb) + self.bhb)

    def output(self, H):
        """
        calculates all outputs for the RNN:

        H is a numpy.ndarray of shape (t, m, 2 * h) that contains the
        concatenated hidden states from both directions, excluding their
        initialized states
            t is the number of time steps
            m is the batch size for the data
            h is the dimensionality of the hidden states
        Returns: Y, the outputs
        """
        def softmax(x):
            max = np.max(x, axis=2, keepdims=True)
            return np.exp(x - max) / np.sum(
                np.exp(x - max), axis=2, keepdims=True)

        return softmax((H @ self.Wy) + self.by)
