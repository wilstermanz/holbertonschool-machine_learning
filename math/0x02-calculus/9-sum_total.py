#!/usr/bin/env python3
"""Contains summation_i_squared() and main test"""


def summation_i_squared(n):
    """return sigma i^2 as i goes to n"""
    if n < 1:
        return None
    return sum(map(lambda x: x**2, range(1, n + 1)))


if __name__ == "__main__":
    n = 5
    print(summation_i_squared(n))
