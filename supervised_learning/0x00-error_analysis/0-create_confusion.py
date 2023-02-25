#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """
    creates a confusion matrix:

    labels is a one-hot numpy.ndarray of shape (m, classes) containing the
    correct labels for each data point
        m is the number of data points
        classes is the number of classes
    logits is a one-hot numpy.ndarray of shape (m, classes) containing the
    predicted labels
    Returns: a confusion numpy.ndarray of shape (classes, classes) with row
    indices representing the correct labels and column indices representing the
    predicted labels
    """
    confusion_matrix = np.zeros((labels.shape[1], labels.shape[1]))
    labels_index = np.where(labels == 1)[1]
    logits_index = np.where(logits == 1)[1]
    for i in range(len(labels)):
        confusion_matrix[labels_index[i]][logits_index[i]] += 1
    return confusion_matrix
