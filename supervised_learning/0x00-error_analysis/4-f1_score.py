#!/usr/bin/env python3
"""Task 4"""
import numpy as np


def f1_score(confusion):
    """
    calculates the F1 score of a confusion matrix:

    * confusion is a confusion numpy.ndarray of shape (classes, classes) where
      row indices represent the correct labels and column indices represent the
      predicted labels
        * classes is the number of classes
    * Returns: a numpy.ndarray of shape (classes,) containing the F1 score of
      each class
    """
    TP = np.diag(confusion)
    FP = np.sum(confusion, axis=0) - TP
    FN = np.sum(confusion, axis=1) - TP
    return (TP + TP) / (TP + TP + FP + FN)
