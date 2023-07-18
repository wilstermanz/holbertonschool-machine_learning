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
    H = []
    Y = []
    H.append(h_0)

    for l, rnn_cell in enumerate(rnn_cells):
        print(l)
        h_t = H[l].copy()

        for data in X:
            if l == 0:
                h_t[l], y_t = rnn_cell.forward(h_t[l], data)
            else:
                h_t[l], y_t = rnn_cell.forward(h_t[l], h_t[l - 1])

        H.append(h_t.copy())

    Y.append(y_t)

    return np.stack(H), np.stack(Y)
