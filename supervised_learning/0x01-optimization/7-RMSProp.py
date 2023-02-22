#!/usr/bin/env python3
"""Task 7"""
import tensorflow as tf
import numpy as np


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm"""
    new_moment = (beta2 * s) + ((grad**2) * (1 - beta2))
    updated_variable = var - (alpha * grad) / np.sqrt(new_moment + epsilon)
    return updated_variable, new_moment
