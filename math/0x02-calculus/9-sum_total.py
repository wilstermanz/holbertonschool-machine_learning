#!/usr/bin/env python3
"""Contains summation_i_squared() and main test"""


def summation_i_squared(n):
    """uses recursion to return sigma i^2 as i goes to n"""
    if n == 1:
        return 1
    return n**2 + summation_i_squared(n - 1)


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
