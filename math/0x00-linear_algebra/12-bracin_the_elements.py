#!/usr/bin/env python3
"""Contains np_elementwise()"""
import numpy as np


def np_transpose(mat1, mat2):
    """
    Performs element-wise addition,
    subtraction, multiplication, and division
    """
    return (np.add(mat1, mat2), np.subtract(mat1, mat2),
            np.multiply(mat1, mat2), np.divide(mat1, mat2))
