#!/usr/bin/env python3
"""Task 6"""
import tensorflow as tf


calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha,
          iterations, save_path="/tmp/model.ckpt"):
    """builds, trains, and saves a neural network classifier"""
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        for i in range(iterations):
            if i % 100 == 0:
                training_cost, training_accuracy = sess.run(
                    (loss, accuracy), feed_dict={x: X_train, y: Y_train})
                valid_cost, valid_accuracy = sess.run(
                    (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
                print("After {} iterations:\n".format(i) +
                      "\tTraining Cost: {}\n".format(training_cost) +
                      "\tTraining Accuracy: {}\n".format(training_accuracy) +
                      "\tValidation Cost: {}\n".format(valid_cost) +
                      "\tValidation Accuracy: {}".format(valid_accuracy))
            sess.run((train_op), feed_dict={x: X_train, y: Y_train})

        training_cost, training_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_train, y: Y_train})
        valid_cost, valid_accuracy = sess.run(
            (loss, accuracy), feed_dict={x: X_valid, y: Y_valid})
        print("After {} iterations:\n".format(iterations) +
              "\tTraining Cost: {}\n".format(training_cost) +
              "\tTraining Accuracy: {}\n".format(training_accuracy) +
              "\tValidation Cost: {}\n".format(valid_cost) +
              "\tValidation Accuracy: {}".format(valid_accuracy))
        saved = tf.train.Saver().save(sess, save_path)
    return saved
