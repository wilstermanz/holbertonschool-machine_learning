#!/usr/bin/env python3
"""Task 24 - One-Hot Encode"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    try:
        one_hot = np.zeros((classes, len(Y)))
        one_hot[Y, np.arange(len(Y))] = 1
        return one_hot
    except Exception:
        return None
