#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def pca(X, ndim):
    """
    performs PCA on a dataset:

    X is a numpy.ndarray of shape (n, d) where:
        n is the number of data points
        d is the number of dimensions in each point
    ndim is the new dimensionality of the transformed X
    Returns: T, a numpy.ndarray of shape (n, ndim) containing the transformed
      version of X

    """
    # Normalize X
    normX = X - np.mean(X, axis=0)

    # SVD of normalized X
    u, s, vh = np.linalg.svd(normX, full_matrices=False)

    # Calculate weights matrix
    W = vh[:ndim].T

    # Transform X using ndim components
    T = normX @ W

    return T


if __name__ == "__main__":
    """Test file"""
    X = np.loadtxt("mnist2500_X.txt")
    print('X:', X.shape)
    print(X)
    T = pca(X, 50)
    print('T:', T.shape)
    print(T)
