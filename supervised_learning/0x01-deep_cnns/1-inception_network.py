#!/usr/bin/env python3
"""Task 0"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    builds the inception network as described in Going Deeper with
      Convolutions (2014):

    You can assume the input data will have shape (224, 224, 3)
    All convolutions inside and outside the inception block should use a
      rectified linear activation (ReLU)
    You may use inception_block =
      __import__('0-inception_block').inception_block
    Returns: the keras model
    """
    # input layer
    input = K.Input(shape=(224, 224, 3))

    # convolutions and pooling
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=2,
        padding='same',
        activation='relu'
        )(input)
    pool1 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        )(conv1)
    conv2 = K.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        padding='valid',
        activation='relu'
        )(pool1)
    conv3 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        strides=1,
        padding='same',
        activation='relu'
        )(conv2)
    pool2 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        )(conv3)

    # inception blocks
    incep1 = inception_block(pool2, [64, 96, 128, 16, 32, 32])
    incep2 = inception_block(incep1, [128, 128, 192, 32, 96, 64])

    # pooling layer
    pool3 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        )(incep2)

    # more inception blocks
    incep3 = inception_block(pool3, [192, 96, 208, 16, 48, 64])
    incep4 = inception_block(incep3, [160, 112, 224, 24, 64, 64])
    incep5 = inception_block(incep4, [128, 128, 256, 24, 64, 64])
    incep6 = inception_block(incep5, [112, 144, 288, 32, 64, 64])
    incep7 = inception_block(incep6, [256, 160, 320, 32, 128, 128])

    # pooling layer
    pool4 = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=2,
        padding='same',
        )(incep7)

    # even more inception blocks
    incep8 = inception_block(pool4, [256, 160, 320, 32, 128, 128])
    incep9 = inception_block(incep8, [384, 192, 384, 48, 128, 128])

    # pooling layer
    pool5 = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=1,
        padding='same'
        )(incep9)

    # dropout layer
    dropout1 = K.layers.Dropout(.4)(pool5)

    # output layer
    output = K.layers.Dense(
        units=1000,
        activation='softmax'
        )(dropout1)

    # build and return model
    return K.Model(input, output)
