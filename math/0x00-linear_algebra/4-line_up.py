#!/usr/bin/env python3
"""Module contains add_arrays functions and main test"""


def add_arrays(arr1, arr2):
    """Adds two arrays elementwise"""
    if len(arr1) != len(arr2):
        return None
    arr3 = []
    for i in range(len(arr1)):
        arr3.append(arr1[i] + arr2[i])
    return arr3


if __name__ == "__main__":
    """main test"""
    add_arrays = __import__('4-line_up').add_arrays

    arr1 = [1, 2, 3, 4]
    arr2 = [5, 6, 7, 8]
    print(add_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
    print(add_arrays(arr1, [1, 2, 3]))
