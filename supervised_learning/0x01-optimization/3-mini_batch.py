#!/usr/bin/env python3
"""Task 3"""
import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """
    trains a loaded neural network model
    using mini-batch gradient descent
    """

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

    with tf.Session() as sess:
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)
        graph = tf.get_default_graph()
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')
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
            if (epoch < epochs):
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)
                for step in range(num_batches):
                    if step < (num_batches - 1):
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
