#!/usr/bin/env python3
"""Task 3"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    return tf.metrics.accuracy(y, y_pred)
