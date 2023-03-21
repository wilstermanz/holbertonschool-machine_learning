#!/usr/bin/env python3
"""Task 2"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    builds an identity block as described in Deep Residual Learning for Image
      Recognition (2015):

        A_prev is the output from the previous layer
        filters is a tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by batch
          normalization along the channels axis and a rectified linear
          activation (ReLU), respectively.
        All weights should use he normal initialization
        Returns: the activated output of the identity block
    """
    # unpack filters
    f11, f3, f12 = filters

    # set initialization
    init = K.initializers.he_normal()

    # layers
    conv1 = K.layers.Conv2D(
        filters=f11,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
        )(A_prev)
    bn1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.ReLU()(bn1)

    conv2 = K.layers.Conv2D(
        filters=f3,
        kernel_size=(3, 3),
        padding='same',
        kernel_initializer=init
        )(act1)
    bn2 = K.layers.BatchNormalization(axis=3)(conv2)
    act2 = K.layers.ReLU()(bn2)

    conv3 = K.layers.Conv2D(
        filters=f12,
        kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
        )(act2)
    bn3 = K.layers.BatchNormalization(axis=3)(conv3)
    act3 = K.layers.ReLU()(bn3)

    add = K.layers.Add()([act3, A_prev])

    return K.layers.ReLU()(add)
