#!/usr/bin/env python3
"""Task 1"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """Creates a layer in a DNN"""
    layer = tf.layers.dense(
        prev,
        n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
        name='layer',
    )
    return layer
