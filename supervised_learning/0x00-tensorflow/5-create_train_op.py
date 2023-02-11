#!/usr/bin/env python3
"""Task 5"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """creates the training operation for the network"""
    optimizer = tf.train.GradientDescentOptimizer(alpha)
    return optimizer.minimize(loss)
