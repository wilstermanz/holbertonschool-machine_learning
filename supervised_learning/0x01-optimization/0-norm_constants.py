#!/usr/bin/env python3
"""Task 0"""
import numpy as np
import tensorflow as tf


def normalization_constants(X):
    """calculates the normalization (standardization) constants of a matrix"""
    return np.mean(X, axis=0), np.std(X, axis=0)
