#!/usr/bin/env python3
"""Task 0 - Neuron"""

import numpy as np


class Neuron:
    """defines a single neuron performing binary classification"""
    def __init__(self, nx):
        """Initializes a new Neuron"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
