#!/usr/bin/env python3
"""Tasks 3-5 - Exponential distribution"""


class Exponential:
    """class to represent an exponential distribution"""

    def __init__(self, data=None, lambtha=1.):
        """Intializes Exponential class by setting data annd lambtha"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = 1 / (sum(data) / len(data))

    def pdf(self, k):
        """Calculates PDF for given time period"""
        e = 2.7182818285
        if k < 0:
            return 0
        return self.lambtha * (e ** (-self.lambtha * k))

    # def cdf(self, k):
    #     """calculates the CDF for given number of successes"""
    #     pass
