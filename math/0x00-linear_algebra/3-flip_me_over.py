#!/usr/bin/env python3
"""Contains matrix_transpose function and main test"""


def matrix_transpose(matrix):
    """Transposes a matrix"""
    output = []
    for i in range(len(matrix[0])):
        output.append([])
        for j in range(len(matrix)):
            output[i].append(matrix[j][i])
    return output


if __name__ == "__main__":
    """main test"""
    mat1 = [[1, 2], [3, 4]]
    print(mat1)
    print(matrix_transpose(mat1))
    mat2 = [[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15],
            [16, 17, 18, 19, 20], [21, 22, 23, 24, 25], [26, 27, 28, 29, 30]]
    print(mat2)
    print(matrix_transpose(mat2))
