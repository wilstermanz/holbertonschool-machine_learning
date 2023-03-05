#!/usr/bin/env python3
"""Task 7"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                save_best=False, filepath=None, verbose=True, shuffle=False):
    """
    saves the best iteration of the model:

    * save_best is a boolean indicating whether to save the model after each
      epoch if it is the best
        * a model is considered the best if its validation loss is the lowest
          that the model has obtained
    * filepath is the file path where the model should be saved
    """
    # Create callbacks list
    callbacks = []

    # Add early stopping
    if validation_data and early_stopping:
        callbacks.append(K.callbacks.EarlyStopping(patience=patience))

    # Add learning rate decay
    if validation_data and learning_rate_decay:
        callbacks.append(K.callbacks.LearningRateScheduler(
            schedule=lambda epoch: alpha / (1 + epoch * decay_rate),
            verbose=True
            ))

    # Save the best model
    if save_best:
        callbacks.append(K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor="val_loss",
            save_best_only=True
        ))

    # Train the model
    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=validation_data,
        verbose=verbose,
        shuffle=shuffle,
        callbacks=callbacks
    )
