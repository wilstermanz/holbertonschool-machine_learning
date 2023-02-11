#!/usr/bin/env python3
"""Task 2"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    a = x
    for i in range(len(layer_sizes)):
        layer = create_layer(a, layer_sizes[i], activations[i])
        a = layer
    return a
