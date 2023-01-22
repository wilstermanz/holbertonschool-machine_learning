#!/usr/bin/env python3
"""Task 0 - Initialize Poisson"""


class Poisson:
    """class to represent a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Intializes Poisson class"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("ambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)
