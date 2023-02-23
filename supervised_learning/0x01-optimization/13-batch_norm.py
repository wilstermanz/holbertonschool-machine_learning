#!/usr/bin/env python3
"""Task 13"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    normalizes an unactivated output of a neural network
    using batch normalization
    """
    mu = np.mean(Z, axis=0)
    sigma = np.std(Z, axis=0)
    Z_norm = (Z - mu) / np.sqrt(sigma ** 2 + epsilon)
    return gamma * Z_norm + beta
