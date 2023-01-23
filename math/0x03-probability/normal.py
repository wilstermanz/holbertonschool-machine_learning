#!/usr/bin/env python3
"""Tasks 6-9 - Normal distribution"""
e = 2.7182818285
pi = 3.1415926536


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
                                        data))) / len(data)) ** .5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """Calculates PDF for given x-value"""
        return (1 / (self.stddev * ((2 * pi) ** .5))) * \
            (e ** (-.5 * (((x - self.mean) / self.stddev) ** 2)))

    def cdf(self, x):
        """calculates the CDF for given x-value"""
        def erf(x):
            return 2 / (pi ** .5) * \
                (x - (x**3 / 3) + (x**5 / 10) - (x**7 / 42) + (x**9 / 216))
        return .5 * (1 + erf((x - self.mean) / (self.stddev * 2**.5)))
