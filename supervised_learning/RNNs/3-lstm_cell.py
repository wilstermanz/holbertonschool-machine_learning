#!/usr/bin/env python3
"""Task 3"""
import numpy as np


class LSTMCell:
    """represents an LSTM unit"""
    def __init__(self, i, h, o):
        """
        i is the dimensionality of the data
        h is the dimensionality of the hidden state
        o is the dimensionality of the outputs
        Creates the public instance attributes Wf, Wu, Wc, Wo, Wy, bf, bu, bc,
        bo, by that represent the weights and biases of the cell
            Wf and bf are for the forget gate
            Wu and bu are for the update gate
            Wc and bc are for the intermediate cell state
            Wo and bo are for the output gate
            Wy and by are for the outputs
        The weights should be initialized using a random normal distribution
        in the order listed above
        The weights will be used on the right side for matrix multiplication
        The biases should be initialized as zeros
        """
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        performs forward propagation for one time step

        x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batche size for the data
        h_prev is a numpy.ndarray of shape (m, h) containing the previous
        hidden state
        c_prev is a numpy.ndarray of shape (m, h) containing the previous cell
        state
        The output of the cell should use a softmax activation function
        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """
        def softmax(x):
            max = np.max(x, axis=1, keepdims=True)
            return np.exp(x - max) / np.sum(
                np.exp(x - max), axis=1, keepdims=True)

        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        h_prev_x_t = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        forget_out = sigmoid((h_prev_x_t @ self.Wf) + self.bf)

        # Update gate
        input_out = sigmoid((h_prev_x_t @ self.Wu) + self.bu)
        intermediate_state = np.tanh((h_prev_x_t @ self.Wc) + self.bc)
        update_out = input_out * intermediate_state

        c_next = c_prev * forget_out + update_out

        # Output gate
        output_out = sigmoid((h_prev_x_t @ self.Wo) + self.bo)
        h_next = output_out * np.tanh(c_next)

        y = softmax((h_next @ self.Wy) + self.by)

        return h_next, c_next, y
