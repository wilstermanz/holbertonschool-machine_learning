#!/usr/bin/env python3

determinant = __import__('0-determinant').determinant

try:
    determinant([[1], [1]])
except ValueError as e:
    print(str(e))
try:
    determinant([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
except ValueError as e:
    print(str(e))
try:
    determinant([[1, 2, 3], [1, 2, 3, 4], [1, 2, 3]])
except ValueError as e:
    print(str(e))
