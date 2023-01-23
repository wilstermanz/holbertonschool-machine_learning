#!/usr/bin/env python3
"""Tasks 10-12 - Binomial distribution"""
e = 2.7182818285
pi = 3.1415926536


class Binomial:
    """Represents a binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """Initializes class instance"""
        self.n = int(n)
        self.p = float(p)
        self.data = data
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            if p < 0 or p > 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            mean = sum(data) / len(data)
            variance = sum(list(map(lambda x: (x - mean) ** 2,
                                    data))) / len(data)
            self.p = 1 - (variance / mean)
            self.n = round(mean / self.p)
            self.p = mean / self.n
