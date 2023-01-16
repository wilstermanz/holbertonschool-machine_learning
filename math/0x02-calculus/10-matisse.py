#!/usr/bin/env python3
"""Contains poly_derivative() and main test"""


def poly_derivative(poly):
    try:
        if len(poly) > 1:
            derivatives = poly[1:]
            for index in range(1, len(derivatives)):
                derivatives[index] *= index + 1
            return derivatives
        elif len(poly) == 1:
            return [0]
        else:
            return None
    except Exception:
        return None



if __name__ == "__main__":
    poly = [5, 3, 0, 1]
    print(poly_derivative(poly))
