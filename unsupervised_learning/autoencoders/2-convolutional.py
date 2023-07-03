#!/usr/bin/env python3
"""Task 2"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    input_dims is a tuple of integers containing the dimensions of the model
    input
    filters is a list containing the number of filters for each convolutional
    layer in the encoder, respectively
        the filters should be reversed for the decoder
    latent_dims is a tuple of integers containing the dimensions of the latent
    space representation
    Each convolution in the encoder should use a kernel size of (3, 3) with
    same padding and relu activation, followed by max pooling of size (2, 2)
    Each convolution in the decoder, except for the last two, should use a
    filter size of (3, 3) with same padding and relu activation, followed by
    upsampling of size (2, 2)
        The second to last convolution should instead use valid padding
        The last convolution should have the same number of filters as the
        number of channels in input_dims with sigmoid activation and no
        upsampling
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    """
    # encoder
    input_img = keras.Input(shape=input_dims)

    hidden = input_img

    for filter in filters:
        hidden = keras.layers.Conv2D(filters=filter,
                                     kernel_size=(3, 3),
                                     padding='same',
                                     activation='relu')(hidden)
        hidden = keras.layers.MaxPooling2D(pool_size=(2, 2),
                                           padding='same')(hidden)

    encoder = keras.Model(input_img, hidden)

    # decoder
    decode_input = keras.Input(shape=latent_dims)

    hidden = decode_input

    for i, filter in enumerate(reversed(filters)):
        padding = 'valid' if i == len(filters) - 1 else 'same'

        hidden = keras.layers.Conv2D(filters=filter,
                                     kernel_size=(3, 3),
                                     padding=padding,
                                     activation='relu')(hidden)
        hidden = keras.layers.UpSampling2D(size=(2, 2))(hidden)

    decoded_output = keras.layers.Conv2D(filters=1,
                                         kernel_size=(3, 3),
                                         padding='same',
                                         activation='sigmoid')(hidden)

    decoder = keras.Model(decode_input, decoded_output)

    encoder_output = encoder(input_img)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(input_img, decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
