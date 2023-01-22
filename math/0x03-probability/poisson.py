#!/usr/bin/env python3
"""Task 0 - Initialize Poisson"""


class Poisson:
    """class to represent a poisson distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Intializes Poisson class by setting data annd lambtha"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data) / len(data)

    def pmf(self, k):
        """Calculates PMF for given number of successes"""
        e = 2.7182818285
        k = int(k)
        if k <= 0:
            return 0
        kFact = 1
        for i in range(1, k + 1):
            kFact *= i
        return (self.lambtha ** k) * ((e ** -self.lambtha) / kFact)

    def cdf(self, k):
        """calculates the CDF for given number of successes"""
        e = 2.7182818285
        k = int(k)
        if k <= 0:
            return 0
        cdf = 0
        for x in range(0, k + 1):
            xFact = 1
            for i in range(1, x + 1):
                xFact *= i
            cdf += (self.lambtha ** x) * ((e ** -self.lambtha) / xFact)
        return cdf
