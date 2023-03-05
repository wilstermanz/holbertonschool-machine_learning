#!/usr/bin/env python3
"""Task 6"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    train the model using early stopping:

    * early_stopping is a boolean that indicates whether early stopping
      should be used
        * early stopping should only be performed if validation_data exists
        * early stopping should be based on validation loss
    * patience is the patience used for early stopping
    """
    if validation_data and early_stopping:
        callbacks = K.callbacks.EarlyStopping(patience=patience)
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=[callbacks]
    )
