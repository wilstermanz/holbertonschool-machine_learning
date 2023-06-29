#!/usr/bin/env python3
"""Task 1"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    creates an autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    All layers should use a relu activation except for the last layer in the
    decoder, which should use sigmoid
    """
    reg = keras.regularizers.L1(lambtha)

    # encoder
    input_img = keras.Input(shape=(input_dims,))

    hidden = keras.layers.Dense(hidden_layers[0], activation='relu')(input_img)

    for layer_size in hidden_layers[1:]:
        hidden = keras.layers.Dense(layer_size, activation='relu')(hidden)

    encoded_output = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=reg)(hidden)

    encoder = keras.Model(input_img, encoded_output)

    # decoder
    decode_input = keras.Input(shape=(latent_dims,))

    hidden = keras.layers.Dense(
        hidden_layers[-1], activation='relu')(decode_input)

    for layer_size in reversed(hidden_layers[:-1]):
        hidden = keras.layers.Dense(layer_size, activation='relu')(hidden)

    decoded_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(hidden)

    decoder = keras.Model(decode_input, decoded_output)

    # autoencoder
    encoder_output = encoder(input_img)
    decoder_output = decoder(encoder_output)
    auto = keras.Model(input_img, decoder_output)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
