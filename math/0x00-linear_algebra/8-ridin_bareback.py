#!/usr/bin/env python3
"""mat_mul() function and main test"""


def mat_mul(mat1, mat2):
    """performs matrix multiplication"""
    if len(mat1[0]) != len(mat2):
        return None

    mat3 = []
    for i in range(len(mat1)):
        mat3.append([])
        for j in range(len(mat2[0])):
            mat3[i].append(0)
            for k in range(len(mat2)):
                mat3[i][j] += mat1[i][k] * mat2[k][j]
    return mat3.copy()


if __name__ == "__main__":
    """Main test"""
    mat1 = [[1, 2],
            [3, 4],
            [5, 6]]
    mat2 = [[1, 2, 3, 4],
            [5, 6, 7, 8]]
    print(mat_mul(mat1, mat2))
