#!/usr/bin/env python3
"""Task 5 - PDF"""
import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution:

    X is a numpy.ndarray of shape (n, d) containing the data points whose PDF
    should be evaluated
    m is a numpy.ndarray of shape (d,) containing the mean of the distribution
    S is a numpy.ndarray of shape (d, d) containing the covariance of the
    distribution
    You are not allowed to use any loops
    You are not allowed to use the function numpy.diag or the method
    numpy.ndarray.diagonal
    Returns: P, or None on failure
        P is a numpy.ndarray of shape (n,) containing the PDF values for
        each data point
    All values in P should have a minimum value of 1e-300
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None

    if type(m) is not np.ndarray or len(m.shape) != 1:
        return None

    if type(S) is not np.ndarray or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d:
        return None

    if S.shape[0] != d or S.shape[1] != d:
        return None

    pass
