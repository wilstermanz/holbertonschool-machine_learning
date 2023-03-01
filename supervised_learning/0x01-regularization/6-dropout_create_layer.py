#!/usr/bin/env python3
"""Task 6"""
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    creates a layer of a neural network using dropout:

        * prev is a tensor containing the output of the previous layer
        * n is the number of nodes the new layer should contain
        * activation is the activation function that should be used on the
          layer
        * keep_prob is the probability that a node will be kept
        * Returns: the output of the new layer
    """
    regularizer = tf.layers.Dropout(1 - keep_prob)
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(n,
                            activation=activation,
                            kernel_initializer=init,
                            kernel_regularizer=regularizer)
    return layer(prev)
