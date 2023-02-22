#!/usr/bin/env python3
"""Task 8"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the RMSProp optimization algorithm
    """
    optimizer = tf.RMSPropOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)
