#!/usr/bin/env python3
"""Contains poly_integral() and main test"""


def poly_integral(poly, C=0):
    """calculates the integral of a polynomial"""
    if type(poly) is not list or type(C) not in [int, float] or len(poly) == 0:
        return None
    coefficients = [C, *poly]
    for i in range(1, len(poly) + 1):
        print(f"{coefficients[i]} / {i}")
        coefficients[i] /= i
    return coefficients


if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_integral(poly))
