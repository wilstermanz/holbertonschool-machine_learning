#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs forward propagation over a pooling layer of a neural network:

        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c_prev)
          containing the output of the previous layer
            m is the number of examples
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
            c_prev is the number of channels in the previous layer
        kernel_shape is a tuple of (kh, kw) containing the size of the kernel
          for the pooling
            kh is the kernel height
            kw is the kernel width
        stride is a tuple of (sh, sw) containing the strides for the pooling
            sh is the stride for the height
            sw is the stride for the width
        mode is a string containing either max or avg, indicating whether to
          perform maximum or average pooling, respectively
        you may import numpy as np
        Returns: the output of the pooling layer
    """
    # set dimensions
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # create output
    output = np.zeros((m,
                      (h - kh) // sh + 1,
                      (w - kw) // sw + 1,
                       c))

    # loop over output
    for x in range(output.shape[2]):
        for y in range(output.shape[1]):
            if mode == 'max':
                output[:, y, x, :] = np.max(
                    A_prev[:,
                           y*sh:y*sh+kh,
                           x*sw:x*sw+kw,
                           :], axis=(1, 2))
            if mode == 'avg':
                output[:, y, x, :] = np.average(
                    A_prev[:,
                           y*sh:y*sh+kh,
                           x*sw:x*sw+kw,
                           :], axis=(1, 2))
    return output
