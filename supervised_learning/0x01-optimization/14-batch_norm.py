#!/usr/bin/env python3
"""Task 14"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer
    for a neural network in tensorflow
    """
    layer = tf.layers.dense(
        prev,
        n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
    )
    mean, variance = tf.nn.moments(layer, 0)
    gamma = tf.Variable(tf.ones(n), True)
    beta = tf.Variable(tf.zeros(n), True)
    epsilon = 10 ** -8
    return activation(tf.nn.batch_normalization(
        layer, mean, variance, beta, gamma, epsilon))
