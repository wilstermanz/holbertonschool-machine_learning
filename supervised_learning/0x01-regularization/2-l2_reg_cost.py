#!/usr/bin/env python3
"""Task 2"""
import tensorflow as tf


def l2_reg_cost(cost):
    """
    calculates the cost of a neural network with L2 regularization:

        * cost is a tensor containing the cost of the network without L2
          regularization
        * Returns: a tensor containing the cost of the network accounting for
          L2 regularization
    """
    return cost + tf.losses.get_regularization_losses()
