#!/usr/bin/env python3
"""Task 1"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library:

    * nx is the number of input features to the network
    * layers is a list containing the number of nodes in each layer of the
      network
    * activations is a list containing the activation functions used for each
      layer of the network
    * lambtha is the L2 regularization parameter
    * keep_prob is the probability that a node will be kept for dropout
    * You are not allowed to use the Input class
    * Returns: the keras model
    """
    # create input layer
    input = K.layers.Input(shape=(nx,))

    # Create hidden layers
    x = input
    for i in range(len(layers) - 1):
        x = K.layers.Dense(
            units=layers[i],
            activation=activations[i],
            kernel_regularizer=K.regularizers.l2(lambtha)
            )(x)
        x = K.layers.Dropout(rate=1 - keep_prob)(x)

    # Create output layer
    output = K.layers.Dense(
        units=layers[-1],
        activation=activations[-1],
        kernel_regularizer=K.regularizers.l2(lambtha)
        )(x)

    return K.Model(input, output)
