#!/usr/bin/env python3
"""Contains np_cat()"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices along a specific axis"""
    return np.concatenate((mat1, mat2), axis)


if __name__ == "__main__":
    mat1 = np.array([[11, 22, 33], [44, 55, 66]])
    mat2 = np.array([[1, 2, 3], [4, 5, 6]])
    mat3 = np.array([[7], [8]])
    print(np_cat(mat1, mat2))
    print(np_cat(mat1, mat2, axis=1))
    print(np_cat(mat1, mat3, axis=1))
