#!/usr/bin/env python3
"""Tasks 6-9 - Normal distribution"""


class Normal:
    """class to represent a normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """Intializes Exponential class by setting data annd lambtha"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum(list(map(lambda x: (x - self.mean) ** 2,
                                        data))) / len(data)) ** (1/2)

    # def pdf(self, k):
    #     """Calculates PDF for given time period"""
    #     e = 2.7182818285
    #     if k < 0:
    #         return 0
    #     return self.lambtha * (e ** (-self.lambtha * k))

    # def cdf(self, k):
    #     """calculates the CDF for given time period"""
    #     e = 2.7182818285
    #     if k < 0:
    #         return 0
    #     return 1 - (e ** (-self.lambtha * k))
