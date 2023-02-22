#!/usr/bin/env python3
"""Task 7"""
import tensorflow as tf


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """updates a variable using the RMSProp optimization algorithm"""
    new_moment = (beta2 * s) + ((grad**2) * (1 - beta2))
    updated_variable = var - (alpha * grad) / (new_moment + epsilon)**(1/2)
    return updated_variable, new_moment
