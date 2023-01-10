#!/usr/bin/env python3
"""cat_arrays function and main test"""


def cat_arrays(arr1, arr2):
    """concatenates two arrays"""
    return arr1 + arr2


if __name__ == "__main__":
    """Main test"""
    arr1 = [1, 2, 3, 4, 5]
    arr2 = [6, 7, 8]
    print(cat_arrays(arr1, arr2))
    print(arr1)
    print(arr2)
