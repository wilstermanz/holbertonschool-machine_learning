#!/usr/bin/env python3
"""Task 3"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    converts a label vector into a one-hot matrix:

    * The last dimension of the one-hot matrix must be the number of classes
    * Returns: the one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
