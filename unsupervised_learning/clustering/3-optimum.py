#!/usr/bin/env python3
"""Task 3 - Optimize K"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    X: numpy.ndarray shape (n, d) containing the data set
    kmin: positive integer containing minimum number of clusters to check for
    kmax: positive integer containing maximum number of clusters to check for
    iterations: positive integer containing the max number of iterations
    Returns results, d_vars or None, None
        results: list containing outputs of the K-means for each cluster size
        d_vars: list containing the difference in variance from the smallest
            cluster size for each cluster size
    """
    if type(X) is not np.ndarray or X.ndim != 2:
        return None, None
    if type(kmin) is not int or kmin < 1:
        return None, None
    if kmax is None:
        kmax = X.shape[0]
    if type(kmax) is not int or kmax <= 0 or kmin >= kmax:
        return None, None
    if type(iterations) is not int or iterations < 1:
        return None, None

    results, d_vars = [], []
    for i in range(kmin, kmax + 1):
        C, clss = kmeans(X, i, iterations)
        if i == kmin:
            var = variance(X, C)
            results.append((C, clss))
            d_vars.append(0.0)
        else:
            results.append((C, clss))
            d_vars.append(var - variance(X, C))

    return results, d_vars
