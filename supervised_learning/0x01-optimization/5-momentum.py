#!/usr/bin/env python3
"""Task 5"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    updates a variable using the gradient descent with
    momentum optimization algorithm
    """
    new_moment = beta1 * v + (grad * (1 - beta1))
    updated_variable = var - (alpha * new_moment)
    return updated_variable, new_moment
