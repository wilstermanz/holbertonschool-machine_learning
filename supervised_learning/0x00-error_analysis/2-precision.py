#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def precision(confusion):
    """
    calculates the precision for each class in a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the precision of each class
    """
    TP = np.diag(confusion)
    sum = np.sum(confusion, axis=0)
    return TP / sum
