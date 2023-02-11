#!/usr/bin/env python3
"""Task 3"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    answer = tf.math.argmax(y, axis=1)
    prediction = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(prediction, answer)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))
