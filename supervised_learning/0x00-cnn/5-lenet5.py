#!/usr/bin/env python3
"""Task 5"""
import tensorflow.keras as K


def lenet5(X):
    """
    builds a modified version of the LeNet-5 architecture using keras:

    X is a K.Input of shape (m, 28, 28, 1) containing the input images for the
      network
        m is the number of images
    The model should consist of the following layers in order:
        Convolutional layer with 6 kernels of shape 5x5 with same padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Convolutional layer with 16 kernels of shape 5x5 with valid padding
        Max pooling layer with kernels of shape 2x2 with 2x2 strides
        Fully connected layer with 120 nodes
        Fully connected layer with 84 nodes
        Fully connected softmax output layer with 10 nodes
    All layers requiring initialization should initialize their kernels with
      the he_normal initialization method
    All hidden layers requiring activation should use the relu activation
      function
    you may import tensorflow.keras as K
    Returns: a K.Model compiled to use Adam optimization
      (with default hyperparameters) and accuracy metrics
    """

    # Set initializer
    init = K.initializers.he_normal()

    # input layer
    input = X

    # Create convolutional and pooling layers
    convolutional_1 = K.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation='relu',
        kernel_initializer=init,
        )(input)

    pooling_1 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
        )(convolutional_1)

    convolutional_2 = K.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation='relu',
        kernel_initializer=init,
        )(pooling_1)

    pooling_2 = K.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
        )(convolutional_2)

    # Flatten convolutions
    flat = K.layers.Flatten()(pooling_2)

    # Dense layers
    dense_1 = K.layers.Dense(
        units=120,
        activation='relu',
        kernel_initializer=init
    )(flat)

    dense_2 = K.layers.Dense(
        units=84,
        activation='relu',
        kernel_initializer=init
    )(dense_1)

    # output layer
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
    )(dense_2)

    # build model
    model = K.Model(input, output)

    model.compile(
        optimizer=K.optimizers.Adam(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
