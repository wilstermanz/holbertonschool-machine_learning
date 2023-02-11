#!/usr/bin/env python3
"""Task 0"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """Creates two placeholders, x and y"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y
