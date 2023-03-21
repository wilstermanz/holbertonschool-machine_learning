#!/usr/bin/env python3
"""Task 5"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    builds a dense block as described in Densely Connected Convolutional
      Networks:

        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        growth_rate is the growth rate for the dense block
        layers is the number of layers in the dense block
        You should use the bottleneck layers used for DenseNet-B
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization and a
          rectified linear activation (ReLU), respectively
        Returns: The concatenated output of each layer within the Dense Block
          and the number of filters within the concatenated outputs,
          respectively
    """
    # set init
    init = K.initializers.he_normal()

    # use loop to create block
    concatenated_output = X
    for layer in range(layers):

        # batch norm and relu activation
        bn1 = K.layers.BatchNormalization(axis=3)(concatenated_output)
        relu1 = K.layers.ReLU()(bn1)

        # bottleneck layer
        conv1 = K.layers.Conv2D(
            filters=4*growth_rate,
            kernel_size=1,
            strides=1,
            padding='same',
            kernel_initializer=init
            )(relu1)

        # batch norm and relu activation
        bn2 = K.layers.BatchNormalization(axis=3)(conv1)
        relu2 = K.layers.ReLU()(bn2)

        # convolutional layer
        conv2 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=3,
            padding='same',
            kernel_initializer=init
            )(relu2)

        # create concatenated output
        concatenated_output = K.layers.concatenate(
            [concatenated_output, conv2])
        nb_filters += growth_rate

    return concatenated_output, nb_filters
