#!/usr/bin/env python3
"""Task 8"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    performs forward propagation for a bidirectional RNN:

    bi_cell is an instance of BidirectionalCell that will be used for the
    forward propagation
    X is the data to be used, given as a numpy.ndarray of shape (t, m, i)
        t is the maximum number of time steps
        m is the batch size
        i is the dimensionality of the data
    h_0 is the initial hidden state in the forward direction, given as a
    numpy.ndarray of shape (m, h)
        h is the dimensionality of the hidden state
    h_t is the initial hidden state in the backward direction, given as a
    numpy.ndarray of shape (m, h)
    Returns: H, Y
        H is a numpy.ndarray containing all of the concatenated hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    h = h_0.shape[1]
    o = bi_cell.Wy.shape[1]

    # initialize outputs
    H_f = np.zeros((t+1, m, h))
    H_b = np.zeros((t+1, m, h))
    Y = np.zeros((h, o))
    H_f[0] = h_0
    H_b[0] = h_t

    for step, data in enumerate(X, start=1):
        H_f[step] = bi_cell.forward(H_f[step - 1], data)

    for step, data in enumerate(np.flip(X, axis=0), start=1):
        H_b[step] = bi_cell.backward(H_b[step - 1], data)

    H = np.concatenate((H_f[1:], np.flip(H_b[1:], axis=0)), axis=2)

    Y = bi_cell.output(H)

    return H, Y
