#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def normalize(X, m, s):
    """normalizes (standardizes) a matrix"""
    return (X - m) / np.sqrt(s**2 + 10**(-8))
