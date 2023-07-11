#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Inputs:
        rnn_cell: an instance of RNNCell that will be used for the forward
        propagation
        X: the data to be used, given as a numpy.ndarray of shape (t, m, i)
            t: maximum number of time steps
            m: batch size
            i: dimensionality of the data
        h_0: initial hidden state, given as a numpy.ndarray of shape (m, h)
            h: dimensionality of the hidden state

    Returns:
        H: a numpy.ndarray containing all of the hidden states
        Y: a numpy.ndarray containing all of the outputs
    """
    H = []
    Y = []
    H.append(h_0)
    for i, data in enumerate(X):
        h_t, y_t = rnn_cell.forward(H[i], data)
        H.append(h_t)
        Y.append(y_t)

    return np.array(H), np.array(Y)
