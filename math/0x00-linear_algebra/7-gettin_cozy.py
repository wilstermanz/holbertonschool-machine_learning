#!/usr/bin/env python3
"""cat_matrices2D() function and main test"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices along a specified axis"""

    mat3 = []

    if axis == 0:
        for row in mat1:
            mat3.append(row.copy())
        for row in mat2:
            mat3.append(row.copy())

    if axis == 1:
        for i in range(len(mat1)):
            mat3.append(mat1[i].copy() + mat2[i].copy())

    return mat3


if __name__ == "__main__":
    """Main test"""

    mat1 = [[1, 2], [3, 4]]
    mat2 = [[5, 6]]
    mat3 = [[7], [8]]
    mat4 = cat_matrices2D(mat1, mat2)
    mat5 = cat_matrices2D(mat1, mat3, axis=1)
    print(mat4)
    print(mat5)
    mat1[0] = [9, 10]
    mat1[1].append(5)
    print(mat1)
    print(mat4)
    print(mat5)
