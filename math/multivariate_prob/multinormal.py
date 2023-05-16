#!/usr/bin/env python3
"""Tasks 2-3"""
import numpy as np


class MultiNormal:
    """
    Represents a Multivariate Normal distribution
    """
    def __init__(self, data):
        """
        data is a numpy.ndarray of shape (d, n) containing the data set:
        n is the number of data points
        d is the number of dimensions in each data point
        If data is not a 2D numpy.ndarray, raise a TypeError with the message
          data must be a 2D numpy.ndarray
        If n is less than 2, raise a ValueError with the message data must
          contain multiple data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')

        self.mean = np.expand_dims(np.mean(data, axis=1), axis=1)
        self.cov = np.matmul((data - self.mean), (data - self.mean).T) \
            / (data.shape[1] - 1)

    def pdf(self, x):
        """
        calculates the PDF at a data point:

        x is a numpy.ndarray of shape (d, 1) containing the data point whose
          PDF should be calculated
            d is the number of dimensions of the Multinomial instance
        If x is not a numpy.ndarray, raise a TypeError with the message x must
          be a numpy.ndarray
        If x is not of shape (d, 1), raise a ValueError with the message x must
          have the shape ({d}, 1)
        Returns the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')
        if len(x.shape) != 2 or x.shape[1] != 1:
            raise ValueError(
                'x must have the shape ({}, 1)'.format(x.shape[0]))

        p1 = (2 * np.pi) ** (-x.shape[0] / 2)
        p2 = np.linalg.det(self.cov) ** (-0.5)
        p3 = np.matmul((x - self.mean).T, np.linalg.inv(self.cov))
        pdf = p1 * p2 * np.exp(-0.5 * np.matmul(p3, (x - self.mean)))

        return np.squeeze(pdf)
