#!/usr/bin/env python3
"""Task 7: Maximization"""
import numpy as np


def maximization(X, g):
    """
    calculates the maximization step in the EM algorithm for a GMM:

    X is a numpy.ndarray of shape (n, d) containing the data set
    g is a numpy.ndarray of shape (k, n) containing the posterior
    probabilities for each data point in each cluster
    You may use at most 1 loop
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the updated priors for
        each cluster
        m is a numpy.ndarray of shape (k, d) containing the updated centroid
        means for each cluster
        S is a numpy.ndarray of shape (k, d, d) containing the updated
        covariance matrices for each cluster
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None

    if type(g) is not np.ndarray or len(g.shape) != 2:
        return None, None, None
    if False in np.isclose(g.sum(axis=0), np.ones((g.shape[1]))):
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None

    pass
