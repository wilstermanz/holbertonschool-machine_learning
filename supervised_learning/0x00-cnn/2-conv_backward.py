#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    performs back propagation over a convolutional layer of a neural network:

        dZ is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
          partial derivatives with respect to the unactivated output of the
          convolutional layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c_new is the number of channels in the output
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        W is a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
          kernels for the convolution
            kh is the filter height
            kw is the filter width
        b is a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
          applied to the convolution
        padding is a string that is either same or valid, indicating the type
          of padding used
        stride is a tuple of (sh, sw) containing the strides for the
          convolution
            sh is the stride for the height
            sw is the stride for the width
        you may import numpy as np
        Returns: the partial derivatives with respect to the previous layer
          (dA_prev), the kernels (dW), and the biases (db), respectively
    """
    # Set dimensions
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # Create empty outputs
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # Set padding
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pw = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1

    # Padding A_prev and dA_prev
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    dA_prev = np.pad(dA_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    # Fill output
    for frame in range(m):
        for x in range(h_new):
            for y in range(w_new):
                for z in range(c_new):
                    dW[:, :, :, z] += np.multiply(A_prev[frame,
                                                         x*sh:x*sh+kh,
                                                         y*sw:y*sw+kw,
                                                         :],
                                                  dZ[frame, x, y, z]
                                                  )
                    dA_prev[frame,
                            x*sh:x*sh+kh,
                            y*sw:y*sw+kw,
                            :] += np.multiply(W[:, :, :, z],
                                              dZ[frame, x, y, z])

    # Remove padding
    if padding == 'same':
        dA_prev = dA_prev[:, ph:-ph, pw:-pw, :]

    return dA_prev, dW, db
