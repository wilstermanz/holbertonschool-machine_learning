#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    builds a transition layer as described in Densely Connected Convolutional
      Networks:

        X is the output from the previous layer
        nb_filters is an integer representing the number of filters in X
        compression is the compression factor for the transition layer
        Your code should implement compression as used in DenseNet-C
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization and a
          rectified linear activation (ReLU), respectively
        Returns: The output of the transition layer and the number of filters
          within the output, respectively
    """
    # set init
    init = K.initializers.he_normal()

    # normalization and activation
    bn1 = K.layers.BatchNormalization(axis=3)(X)
    relu1 = K.layers.ReLU()(bn1)

    # compression layer
    conv1 = K.layers.Conv2D(
        filters=int(compression*nb_filters),
        kernel_size=1,
        strides=1,
        padding='same',
        kernel_initializer=init
        )(relu1)

    # pooling
    pool1 = K.layers.AveragePooling2D(
        pool_size=2,
        strides=1,
        padding='same'
        )(conv1)

    return pool1, int(compression*nb_filters)
