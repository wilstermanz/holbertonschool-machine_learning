#!/usr/bin/env python3
"""Task 4 - Initialize GMM"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """
    initializes variables for a Gaussian Mixture Model:

    X is a numpy.ndarray of shape (n, d) containing the data set
    k is a positive integer containing the number of clusters
    You are not allowed to use any loops
    Returns: pi, m, S, or None, None, None on failure
        pi is a numpy.ndarray of shape (k,) containing the priors for each
        cluster, initialized evenly
        m is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster, initialized with K-means
        S is a numpy.ndarray of shape (k, d, d) containing the covariance
        matrices for each cluster, initialized as identity matrices
    """

    if type(k) is not int or k < 1:
        return None, None, None

    pi = np.ones((k,)) / k

    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None

    S = np.zeros((k, X.shape[1], X.shape[1]))
    S[:] = np.eye(X.shape[1])

    return pi, m, S
