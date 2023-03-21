#!/usr/bin/env python3
"""Task 7"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    builds the DenseNet-121 architecture as described in Densely Connected
      Convolutional Networks:

        growth_rate is the growth rate
        compression is the compression factor
        You can assume the input data will have shape (224, 224, 3)
        All convolutions should be preceded by Batch Normalization and a
          rectified linear activation (ReLU), respectively
        All weights should use he normal initialization

        Returns: the keras model
    """
    # set nb_filters
    nb_filters = 2 * growth_rate

    # set initialization
    init = K.initializers.he_normal()

    # input layer
    input = K.Input(shape=(224, 224, 3))

    # normalization and activation
    bn1 = K.layers.BatchNormalization(axis=3)(input)
    relu1 = K.layers.ReLU()(bn1)

    # convolutional layer
    conv1 = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=7,
        strides=2,
        padding='same',
        kernel_initializer=init
        )(relu1)

    # pooling layer
    pool1 = K.layers.MaxPool2D(
        pool_size=3,
        strides=2,
        padding='same'
        )(conv1)

    # dense block 1
    db1, nb_filters = dense_block(pool1, nb_filters, growth_rate, 6)

    # transition 1
    trans1, nb_filters = transition_layer(db1, nb_filters, compression)

    # dense block 2
    db2, nb_filters = dense_block(trans1, nb_filters, growth_rate, 12)

    # transition 2
    trans2, nb_filters = transition_layer(db2, nb_filters, compression)

    # dense block 3
    db3, nb_filters = dense_block(trans2, nb_filters, growth_rate, 24)

    # transition 3
    trans3, nb_filters = transition_layer(db3, nb_filters, compression)

    # dense block 4
    db4, nb_filters = dense_block(trans3, nb_filters, growth_rate, 16)

    # pooling layer
    pool2 = K.layers.AveragePooling2D(
        pool_size=7,
        strides=1,
        padding='same'
        )(db4)

    # output layer
    output = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=init
        )(pool2)

    return K.Model(input, output)
