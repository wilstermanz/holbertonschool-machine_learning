#!/usr/bin/env python3
import numpy as np
"""Task 0"""


def pca(X, var=0.95):
    """
    performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
        all dimensions have a mean of 0 across all data points
    var is the fraction of the variance that the PCA transformation should
      maintain
    Returns: the weights matrix, W, that maintains var fraction of X's
      original variance
    W is a numpy.ndarray of shape (d, nd) where nd is the new dimensionality
      of the transformed X
    """
    # Calculate SVD
    u, s, vh = np.linalg.svd(X, full_matrices=False)

    # Calculate cumulative Explained Variance Ratio
    evr = np.cumsum(s**2) / np.sum(s**2)

    # Find number of principal components
    dims = np.argmax(evr >= var) + 1

    # Calculate Weights matrix
    W = vh[:dims + 1].T

    return W


if __name__ == "__main__":
    np.random.seed(0)
    a = np.random.normal(size=50)
    b = np.random.normal(size=50)
    c = np.random.normal(size=50)
    d = 2 * a
    e = -5 * b
    f = 10 * c

    X = np.array([a, b, c, d, e, f]).T
    m = X.shape[0]
    X_m = X - np.mean(X, axis=0)
    W = pca(X_m)
    T = np.matmul(X_m, W)
    print(T)
    X_t = np.matmul(T, W.T)
    print(np.sum(np.square(X_m - X_t)) / m)
