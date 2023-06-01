#!/usr/bin/env python3
"""Task 1 - K-means"""
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


def kmeans(X, k, iterations=1000):
    """
    performs K-means on a dataset:

    X is a numpy.ndarray of shape (n, d) containing the dataset
        n is the number of data points
        d is the number of dimensions for each data point
    k is a positive integer containing the number of clusters
    iterations is a positive integer containing the maximum number of
    iterations that should be performed
    If no change in the cluster centroids occurs between iterations, your
    function should return
    Initialize the cluster centroids using a multivariate uniform distribution
    (based on 0-initialize.py)
    If a cluster contains no data points during the update step, reinitialize
    its centroid
    You should use numpy.random.uniform exactly twice
    You may use at most 2 loops
    Returns: C, clss, or None, None on failure
        C is a numpy.ndarray of shape (k, d) containing the centroid means for
        each cluster
        clss is a numpy.ndarray of shape (n,) containing the index of the
        cluster in C that each data point belongs to

    """

    # initialize cluster centroids
    C = initialize(X, k)

    # checks
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(iterations) is not int or iterations <= 0:
        return None, None
    if C is None:
        return None, None

    n, d = X.shape
    clss = np.argmin(
        np.linalg.norm(X[:, np.newaxis] - C, axis=2), axis=1)

    for _ in range(iterations):
        new_C = C.copy()
        for i in range(len(C)):
            if len(X[clss == i] > 0):
                new_C[i] = np.mean(X[clss == i], axis=0)
            else:
                new_C[i] = initialize(X, 1)
        clss = np.argmin(
            np.linalg.norm(X[:, np.newaxis] - new_C, axis=2), axis=1)
        if np.array_equal(C, new_C):
            break
        C = new_C

    return C, clss
