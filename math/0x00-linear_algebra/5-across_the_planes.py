#!/usr/bin/env python3
"""Contains add_matrices2D function and main test"""


def add_matrices2D(mat1, mat2):
    """Adds two matrices element-wise"""

    # Check that mat1 and mat2 are the same shape
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None

    # Create output matrix
    mat3 = []

    # Add matrices
    for i in range(len(mat1)):
        mat3.append([])     # add row to mat3 for each row in mat1
        for j in range(len(mat1[i])):
            mat3[i].append(mat1[i][j] + mat2[i][j])

    # return output matrix
    return mat3


if __name__ == "__main__":
    """Main test"""
    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6], [7, 8]]
    print(add_matrices2D(mat1, mat2))
    print(mat1)
    print(mat2)
    print(add_matrices2D(mat1, [[1, 2, 3], [4, 5, 6]]))
