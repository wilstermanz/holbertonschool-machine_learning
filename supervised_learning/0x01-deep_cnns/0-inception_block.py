#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going Deeper with Convolutions
      (2014):

        A_prev is the output from the previous layer
        filters is a tuple or list containing F1, F3R, F3,F5R, F5, FPP,
          respectively:
            F1 is the number of filters in the 1x1 convolution
            F3R is the number of filters in the 1x1 convolution before the 3x3
              convolution
            F3 is the number of filters in the 3x3 convolution
            F5R is the number of filters in the 1x1 convolution before the 5x5
              convolution
            F5 is the number of filters in the 5x5 convolution
            FPP is the number of filters in the 1x1 convolution after the max
              pooling
        All convolutions inside the inception block should use a rectified
          linear activation (ReLU)
        Returns: the concatenated output of the inception block
    """
    # unpack filters
    f1, f3r, f3, f5r, f5, fpp = filters

    # 1x1 convolution inception module
    output_1 = K.layers.Conv2D(
        f1,
        (1, 1),
        padding='same',
        activation='relu'
        )(A_prev)

    # 3x3 convolution inception module
    conv1 = K.layers.Conv2D(
        f3r,
        (1, 1),
        padding='same',
        activation='relu'
        )(A_prev)
    output_2 = K.layers.Conv2D(
        f3,
        (3, 3),
        padding='same',
        activation='relu'
        )(conv1)

    # 5x5 convolution inception module
    conv2 = K.layers.Conv2D(
        f5r,
        (1, 1),
        padding='same',
        activation='relu'
        )(A_prev)
    output_3 = K.layers.Conv2D(
        f5,
        (5, 5),
        padding='same',
        activation='relu'
        )(conv2)

    # 1x1 convolution inception module with 3x3 max pooling
    pool1 = K.layers.MaxPool2D(
        (3, 3),
        strides=1,
        padding='same'
        )(A_prev)
    output_4 = K.layers.Conv2D(
        fpp,
        (1, 1),
        padding='same',
        activation='relu'
        )(pool1)

    # output
    return K.layers.Concatenate()([output_1, output_2, output_3, output_4])
