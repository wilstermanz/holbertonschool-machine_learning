#!/usr/bin/env python3
"""Task 3"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    creates a variational autoencoder:

    input_dims is an integer containing the dimensions of the model input
    hidden_layers is a list containing the number of nodes for each hidden
    layer in the encoder, respectively
        the hidden layers should be reversed for the decoder
    latent_dims is an integer containing the dimensions of the latent space
    representation
    Returns: encoder, decoder, auto
        encoder is the encoder model, which should output the latent
        representation, the mean, and the log variance, respectively
        decoder is the decoder model
        auto is the full autoencoder model
    The autoencoder model should be compiled using adam optimization and binary
    cross-entropy loss
    All layers should use a relu activation except for the mean and log
    variance layers in the encoder, which should use None, and the last layer
    in the decoder, which should use sigmoid
    """
    # encoder
    encoder_input = keras.Input(shape=(input_dims,))

    hidden = encoder_input

    for nodes in hidden_layers:
        hidden = keras.layers.Dense(nodes, activation='relu')(hidden)

    latent_mean = keras.layers.Dense(latent_dims, activation=None)(hidden)
    latent_log_var = keras.layers.Dense(latent_dims, activation=None)(hidden)

    def sampling(args):
        z_mean, z_log_var = args
        batch = keras.backend.shape(z_mean)[0]
        dim = keras.backend.shape(z_mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return z_mean + keras.backend.exp(0.5 * z_log_var) * epsilon

    latent_space = keras.layers.Lambda(sampling)([latent_mean, latent_log_var])

    encoder = keras.Model(encoder_input,
                          [latent_space, latent_mean, latent_log_var])

    # decoder
    decoder_input = keras.Input(shape=(latent_dims,))

    hidden = decoder_input

    for nodes in reversed(hidden_layers):
        hidden = keras.layers.Dense(nodes, activation='relu')(hidden)

    decoder_output = keras.layers.Dense(
        input_dims, activation='sigmoid')(hidden)

    decoder = keras.Model(decoder_input, decoder_output)

    # autoencoder
    encoded_output = encoder(encoder_input)
    decoded_output = decoder(encoded_output[0])
    auto = keras.Model(encoder_input, decoded_output)

    def vae_loss(input_img, output):
        # compute the average MSE error, then scale it up i.e. simply sum on
        # all axes
        reconstruction_loss = keras.backend.sum(
            keras.backend.square(output-input_img))
        # compute the KL loss
        kl_loss = -0.5 * keras.backend.sum(
            1 + latent_log_var - keras.backend.square(
                latent_mean) - keras.backend.square(
                    keras.backend.exp(latent_log_var)), axis=-1)
        # return the average loss over all images in batch
        total_loss = keras.backend.mean(reconstruction_loss + kl_loss)
        return total_loss

    auto.compile(optimizer='adam', loss=vae_loss)

    return encoder, decoder, auto