#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    that builds a neural network with the Keras library:

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

    # Create model
    model = K.Sequential()

    # Create input layer
    model.add(K.layers.Input(shape=(nx,)))

    # Create hidden layers
    for i in range(len(layers)):
        model.add(K.layers.Dense(layers[i], activation=activations[i],
                                 kernel_regularizer=K.regularizers.l2(lambtha),
                                 )
                  )
        if i != len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
