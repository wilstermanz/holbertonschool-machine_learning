#!/usr/bin/env python3
"""Task 25 - One-Hot decode"""
import numpy as np


def one_hot_decode(one_hot):
    """converts a one-hot matrix into a vector of labels"""
    try:
        if type(one_hot) is not np.ndarray:
            raise TypeError
        if one_hot.ndim != 2:
            raise ValueError
        return np.argmax(one_hot.T, axis=1)
    except Exception:
        return None
