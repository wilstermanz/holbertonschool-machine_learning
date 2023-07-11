#!/usr/bin/env python3
"""Task 2"""
import numpy as np


class GRUCell:
    """represents a gated recurrent unit"""
    def __init__(self, i, h, o):
        """
        Inputs:
            i is the dimensionality of the data
            h is the dimensionality of the hidden state
            o is the dimensionality of the outputs

        Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by
        that represent the weights and biases of the cell
            Wz and bz are for the update gate
            Wr and br are for the reset gate
            Wh and bh are for the intermediate hidden state
            Wy and by are for the output
        The weights should be initialized using a random normal distribution in
        the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data
        input for the cell
            m is the batch size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        def softmax(x):
            max = np.max(x, axis=1, keepdims=True)
            return np.exp(x - max) / np.sum(
                np.exp(x - max), axis=1, keepdims=True)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        hx = np.concatenate((h_prev, x_t), axis=1)

        # reset gate
        reset_out = sigmoid((hx @ self.Wr) + self.br) * h_prev

        # update gate
        a_hx = sigmoid((hx @ self.Wz) + self.bz)
        update_out = h_prev * (1 - a_hx)

        # intermediate hidden state
        a_ih = np.tanh(
            (np.concatenate((x_t, reset_out), axis=1) @ self.Wh) + self.bh)
        h_next = (a_ih * a_hx) + update_out

        y = softmax((h_next @ self.Wy) + self.by)

        return h_next, y
