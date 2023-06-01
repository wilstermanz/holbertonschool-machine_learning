#!/usr/bin/env python3
"""Task 0 - Initialize"""
import numpy as np


def initialize(X, k):
    """
    initializes cluster centroids for K-means:

    X is a numpy.ndarray of shape (n, d) containing the dataset that will be
    used for K-means clustering
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    The cluster centroids should be initialized with a multivariate uniform
    distribution along each dimension in d:
        The minimum values for the distribution should be the minimum values
        of X along each dimension in d
        The maximum values for the distribution should be the maximum values
        of X along each dimension in d
        You should use numpy.random.uniform exactly once
    You are not allowed to use any loops
    Returns: a numpy.ndarray of shape (k, d) containing the initialized
    centroids for each cluster, or None on failure
    """
    try:
        if k <= 0:
            return None

        min = np.min(X, axis=0)
        max = np.max(X, axis=0)
        centroids = np.random.uniform(low=min, high=max, size=(k, X.shape[1]))
        return centroids

    except Exception:
        return None
