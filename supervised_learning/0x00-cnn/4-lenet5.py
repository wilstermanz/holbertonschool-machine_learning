#!/usr/bin/env python3
"""Task 4"""
import tensorflow as tf


def lenet5(x, y):
    """
    builds a modified version of the LeNet-5 architecture using tensorflow:

        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
          images for the network
            m is the number of images
        y is a tf.placeholder of shape (m, 10) containing the one-hot labels
          for the network
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes
        All layers requiring initialization should initialize their kernels
          with the he_normal initialization method:
          tf.contrib.layers.variance_scaling_initializer()
        All hidden layers requiring activation should use the relu activation
          function
        you may import tensorflow as tf
        you may NOT use tf.keras
        Returns:
            a tensor for the softmax activated output
            a training operation that utilizes Adam optimization (with default
              hyperparameters)
            a tensor for the loss of the netowrk
            a tensor for the accuracy of the network
    """

    # he-norm initialization
    init = tf.contrib.layers.variance_scaling_initializer()
    activation = 'relu'

    # Convolution and pooling layers
    convolutional_1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=activation,
        kernel_initializer=init
        )(x)

    max_pooling_1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
        )(convolutional_1)

    convolutional_2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=activation,
        kernel_initializer=init
        )(max_pooling_1)

    max_pooling_2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)
        )(convolutional_2)

    # Flatten convolutional layers
    flat = tf.layers.Flatten()(max_pooling_2)

    # Dense layers
    dense_1 = tf.layers.Dense(
        units=120,
        activation=activation,
        kernel_initializer=init,
        )(flat)

    dense_2 = tf.layers.Dense(
        units=84,
        activation=activation,
        kernel_initializer=init
        )(dense_1)

    y_pred = tf.layers.Dense(
        units=10,
        kernel_initializer=init
        )(dense_2)

    # loss function
    loss = tf.losses.softmax_cross_entropy(y, y_pred)

    # optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)

    # calculate accuracy
    prediction = tf.math.argmax(y_pred, axis=1)
    correct = tf.math.argmax(y, axis=1)
    accuracy = tf.math.reduce_mean(
        tf.cast(
            tf.math.equal(prediction, correct),
            tf.float32
            )
        )

    # calculate softmax
    softmax = tf.nn.softmax(y_pred)

    return softmax, optimizer, loss, accuracy
