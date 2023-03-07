#!/usr/bin/env python3
"""Task 11"""
import tensorflow.keras as K


def save_config(network, filename):
    """
    saves a model’s configuration in JSON format:

        network is the model whose configuration should be saved
        filename is the path of the file that the configuration should be saved
          to
        Returns: None
    """
    with open(filename, 'w') as f:
        f.write(network.to_json())


def load_config(filename):
    """
    loads a model with a specific configuration:

        filename is the path of the file containing the model’s configuration
          in JSON format
        Returns: the loaded model
    """
    with open(filename, 'r') as f:
        return K.models.model_from_json(f.read())
