#!/usr/bin/env python3
"""Tasks 16 - 23 for Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initializes an instance of DeepNeuralNetwork"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")
            
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        prev = nx
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], prev) * np.sqrt(2 / prev)
            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]
