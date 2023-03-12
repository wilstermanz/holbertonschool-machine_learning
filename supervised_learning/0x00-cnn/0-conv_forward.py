#!/usr/bin/env python3
"""Task 0"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    performs forward propagation over a convolutional layer of a neural
      network:

        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
          kernels for the convolution
            kh is the filter height
            kw is the filter width
            c_prev is the number of channels in the previous layer
            c_new is the number of channels in the output
        b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
          applied to the convolution
        activation is an activation function applied to the convolution
        padding is a string that is either same or valid, indicating the type
          of padding used
        stride is a tuple of (sh, sw) containing the strides for the
          convolution
            sh is the stride for the height
            sw is the stride for the width
        you may import numpy as np
        Returns: the output of the convolutional layer
    """
    # input dimensions
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # get padding
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2

    # pad input
    images_padded = np.pad(A_prev, ((0, 0),
                                    (ph, ph),
                                    (pw, pw),
                                    (0, 0)))

    # create empty output
    output_h = (h_prev + (2 * ph) - kh) // sh + 1
    output_w = (w_prev + (2 * pw) - kw) // sw + 1
    output = np.zeros((m, output_h, output_w, c_new))

    # fill output
    for x in range(output_w):
        for y in range(output_h):
            for z in range(c_new):
                output[:, y, x, z] = np.sum(
                    W[:, :, :, z] * images_padded[:,
                                                  y*sh:y*sh+kh,
                                                  x*sw:x*sw+kw,
                                                  0:c_new
                                                  ],
                    axis=(1, 2, 3)) + b[:, :, :, z]
    return activation(output)
