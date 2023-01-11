#!/usr/bin/env python3
"""Contains np_elementwise()"""


def np_transpose(mat1, mat2):
    """
    Performs element-wise addition,
    subtraction, multiplication, and division
    """
    return (mat1 + mat2, mat1 - mat2,
            mat1 * mat2, mat1 / mat2)
