#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def sensitivity(confusion):
    """
    calculates the sensitivity for each class in a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the sensitivity
      of each class
    """
    TP = np.diag(confusion)
    sum = np.sum(confusion, axis=1)
    return TP / sum
