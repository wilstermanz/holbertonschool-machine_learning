#!/usr/bin/env python3
"""Task 4"""


def determinant(matrix):
    """
    matrix is a list of lists whose determinant should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message matrix must
      be a square matrix
    The list [[]] represents a 0x0 matrix
    Returns: the determinant of matrix
    """
    # type checking
    if matrix == [[]]:
        return 1
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(matrix[i]) != len(matrix):
            raise ValueError('matrix must be non-empty square matrix')

    # determinant for 1x1 matrix
    if len(matrix) == 1:
        return matrix[0][0]

    # determinant for 2x2 matrix
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    # recusion for larger matrices
    det = 0
    for i, n in enumerate(matrix[0]):
        rows = [row for row in matrix[1:]]
        new_m = [row[:i] + row[i + 1:] for row in rows]
        det += n * (-1) ** i * determinant(new_m)
    return det


def minor(matrix):
    """
    calculates the minor matrix of a matrix:

    matrix is a list of lists whose minor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
      matrix must be a non-empty square matrix
    Returns: the minor matrix of matrix
    """
    # checks
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for i in range(len(matrix)):
        if type(matrix[i]) is not list:
            raise TypeError('matrix must be a list of lists')
        if len(matrix[i]) != len(matrix) or len(matrix[0]) == 0:
            raise ValueError('matrix must be a non-empty square matrix')

    # calculate minor
    minor = []
    for i in range(len(matrix)):
        row = []
        for j in range(len(matrix[0])):
            new_rows = [n for n in matrix[:i] + matrix[i + 1:]]
            new_m = [new_row[:j] + new_row[j + 1:] for new_row in new_rows]
            if new_m == []:
                new_m = [[]]
            row.append(determinant(new_m))
        minor.append(row)
    return minor


def cofactor(matrix):
    """
    calculates the cofactor matrix of a matrix:

    matrix is a list of lists whose cofactor matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
      message matrix must be a non-empty square matrix
    Returns: the cofactor matrix of matrix
    """
    cof = minor(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            cof[i][j] *= (-1)**(i+j)

    return cof


def adjugate(matrix):
    """
    adjugate matrix of a matrix:

    matrix is a list of lists whose adjugate matrix should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
      matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the
      message matrix must be a non-empty square matrix
    Returns: the adjugate matrix of matrix
    """
    cof = cofactor(matrix)
    return [list(row) for row in list(zip(*cof))]


def inverse(matrix):
    """
    calculates the inverse of a matrix:

    matrix is a list of lists whose inverse should be calculated
    If matrix is not a list of lists, raise a TypeError with the message
       matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError with the message
       matrix must be a non-empty square matrix
    Returns: the inverse of matrix, or None if matrix is singular
    """
    det = determinant(matrix)
    if det == 0:
        return None
    return [[n / det for n in row] for row in adjugate(matrix)]
