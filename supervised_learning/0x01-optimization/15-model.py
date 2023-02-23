#!/usr/bin/env python3
"""Task 15"""
import numpy as np
import tensorflow as tf


def shuffle_data(X, Y):
    p = np.random.permutation(X.shape[0])
    return X[p], Y[p]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    creates the training operation for a neural network in tensorflow
    using the Adam optimization algorithm
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta2, epsilon=epsilon)
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    creates a learning rate decay operation in tensorflow
    using inverse time decay
    """
    return tf.train.inverse_time_decay(alpha, global_step,
                                       decay_step, decay_rate, True)


def create_batch_norm_layer(prev, n, activation):
    """
    creates a batch normalization layer
    for a neural network in tensorflow
    """
    layer = tf.layers.dense(
        prev,
        n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"),
    )
    mean, variance = tf.nn.moments(layer, 0)
    gamma = tf.Variable(tf.ones(n), True)
    beta = tf.Variable(tf.zeros(n), True)
    epsilon = 10 ** -8
    if activation:
        return activation(tf.nn.batch_normalization(layer, mean, variance,
                                                    beta, gamma, epsilon))
    return tf.nn.batch_normalization(layer, mean, variance,
                                     beta, gamma, epsilon)


def create_placeholders(nx, classes):
    """Creates two placeholders, x and y"""
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")
    return x, y


def print_epoch(epochs, train_cost, train_accuracy,
                valid_cost, valid_accuracy):
    """Prints epoch info"""
    print("After {} epochs:\n".format(epochs) +
          "\tTraining Cost: {}\n".format(train_cost) +
          "\tTraining Accuracy: {}\n".format(train_accuracy) +
          "\tValidation Cost: {}\n".format(valid_cost) +
          "\tValidation Accuracy: {}".format(valid_accuracy))


def print_step(step_number, step_cost, step_accuracy):
    """Prints step info"""
    print("\tStep {}:\n".format(step_number) +
          "\t\tCost: {}\n".format(step_cost) +
          "\t\tAccuracy: {}".format(step_accuracy))


def calculate_accuracy(y, y_pred):
    """calculates the accuracy of a prediction"""
    answer = tf.math.argmax(y, axis=1)
    prediction = tf.math.argmax(y_pred, axis=1)
    equality = tf.math.equal(prediction, answer)
    return tf.math.reduce_mean(tf.cast(equality, tf.float32))


def calculate_loss(y, y_pred):
    """calculates the softmax cross-entropy loss of a prediction"""
    return tf.losses.softmax_cross_entropy(y, y_pred)


def forward_prop(x, layer_sizes=[], activations=[]):
    """creates the forward propagation graph for the neural network"""
    a = x
    for i in range(len(layer_sizes)):
        layer = create_batch_norm_layer(a, layer_sizes[i], activations[i])
        a = layer
    return a


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    builds, trains, and saves a neural network model in tensorflow using
    Adam optimization, mini-batch gradient descent, learning rate decay,
    and batch normalization
    """

    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    x, y = create_placeholders(Data_train[0].shape[1],
                               Data_train[1].shape[1])
    y_pred = forward_prop(x, layers, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        num_batches = (len(X_train) // batch_size)
        if num_batches % batch_size != 0:
            num_batches += 1
        for epoch in range(epochs + 1):
            train_cost, train_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x: X_train,
                                                             y: Y_train})
            valid_cost, valid_accuracy = sess.run((loss, accuracy),
                                                  feed_dict={x: X_valid,
                                                             y: Y_valid})
            print_epoch(epoch, train_cost, train_accuracy,
                        valid_cost, valid_accuracy)
            if epoch < epochs:
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for step in range(num_batches):
                    X_batch = X_shuffled[step * batch_size:
                                         (step + 1) * batch_size]
                    Y_batch = Y_shuffled[step * batch_size:
                                         (step + 1) * batch_size]
                    sess.run(
                        train_op,
                        feed_dict={x: X_batch, y: Y_batch}
                        )
                    if step > 0 and (step + 1) % 100 == 0:
                        step_cost, step_accuracy = sess.run(
                            (loss, accuracy),
                            feed_dict={x: X_batch, y: Y_batch}
                            )
                        print_step(step + 1, step_cost, step_accuracy)
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
