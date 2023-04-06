#!/usr/bin/env python3
import tensorflow.keras as K


def preprocess_data(X, Y):
    """
    Returns unpacked and preprocessed data
    """
    X_p, Y_p = X, K.utils.to_categorical(Y, num_classes=10)
    return X_p, Y_p


def build_model(init=K.initializers.he_normal(),
                activation='relu',
                keep_rate=.5):
    """
    Returns compiled model
    """
    # create input layer
    input = K.layers.Input(shape=(32, 32, 3))

    # resizing layer
    resize = K.layers.Resizing(
        height=224,
        width=224,
        interpolation="bilinear",
        crop_to_aspect_ratio=False)(input)

    # EfficientNet
    eNet = K.applications.EfficientNetV2B2(
        weights='imagenet',
        input_shape=(224, 224, 3),
        include_top=False,
        )
    eNet.trainable = False

    # Freeze previous layers
    frozen_layers = eNet(resize)

    # Reduce features
    conv = K.layers.Conv2D(
        filters=250,
        kernel_size=1,
        strides=1,
        padding='valid',
        activation=activation,
        kernel_initializer=init
        )(frozen_layers)

    # pooling
    pool = K.layers.AvgPool2D(
        pool_size=7,
        strides=1,
        padding='valid',
        )(conv)

    # flatten layer
    flat = K.layers.Flatten()(pool)

    # dropout
    drop = K.layers.Dropout(1 - keep_rate)(flat)

    # output
    output = K.layers.Dense(
        units=10,
        activation='softmax',
        kernel_initializer=init
        )(drop)
    
    # build model
    model = K.models.Model(input, output)

    # compile model
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    # download and preprocess data
    (x_train, y_train), (x_valid, y_valid) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_valid, y_valid = preprocess_data(x_valid, y_valid)

    # define hyperparameters
    init = K.initializers.he_normal()
    activation = 'relu'
    patience = 3
    batch_size = 32
    epochs = 100
    alpha = 0.1
    decay_rate = 1
    keep_rate = .7

    # create model

    model = build_model(init, activation, keep_rate)

    model.summary()

    # callbacks
    callbacks = [K.callbacks.EarlyStopping(patience=patience),
                #  K.callbacks.LearningRateScheduler(
                #     schedule=lambda epoch: alpha / (1 + epoch * decay_rate),
                #     verbose=True),
                 K.callbacks.ModelCheckpoint(
                    filepath='cifar10.h5',
                    monitor='val_loss',
                    save_best_only=True)
                 ]

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(x_valid, y_valid), callbacks=callbacks)
