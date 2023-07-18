#!/usr/bin/env python3
"""Task 4"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    rnn_cells is a list of RNNCell instances of length l that will be used for
    the forward propagation
        l is the number of layers
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state, given as a numpy.ndarray of shape (l, m,
    h)
        h is the dimensionality of the hidden state
    Returns: H, Y
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    # get some dimensions
    t, m, _ = X.shape
    layers, _, h = h_0.shape

    # Initialize H and Y with zeroes
    H = np.zeros((t+1, layers, m, h))
    Y = np.zeros((t, m, rnn_cells[-1].Wy.shape[1]))

    H[0] = h_0
    for step, data in enumerate(X, start=1):
        # forward prop first cell to initialize h_t and y_t
        h_t, y_t = rnn_cells[0].forward(H[step - 1, 0], data)
        H[step, 0] = h_t

        # forward prop remaining cells
        for layer, rnn_cell in enumerate(rnn_cells[1:], start=1):
            # h_prev is hidden state from previous step
            # x_t is h_t from previous layer
            h_t, y_t = rnn_cell.forward(H[step - 1, layer], h_t)

            # Update H for current step and layer
            H[step, layer] = h_t

        # Update Y
        Y[step - 1] = y_t

    return H, Y
