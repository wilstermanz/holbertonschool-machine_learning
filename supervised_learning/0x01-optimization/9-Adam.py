#!/usr/bin/env python3
"""Task 9"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """updates a variable in place using the Adam optimization algorithm"""
    new_v = beta1 * v + (grad * (1 - beta1))
    new_s = beta2 * s + (grad**2 * (1 - beta2))
    v_corrected = new_v / (1 - beta1**t)
    s_corrected = new_s / (1 - beta2**t)
    var = var - alpha * (v_corrected / np.sqrt(s_corrected) + epsilon)
    return var, new_v, new_s
