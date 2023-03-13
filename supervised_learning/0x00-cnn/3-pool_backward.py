#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

        dA is a numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
          partial derivatives with respect to the output of the pooling layer
            m is the number of examples
            h_new is the height of the output
            w_new is the width of the output
            c is the number of channels
        A_prev is a numpy.ndarray of shape (m, h_prev, w_prev, c) containing
          the output of the previous layer
            h_prev is the height of the previous layer
            w_prev is the width of the previous layer
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
        Returns: the partial derivatives with respect to the previous layer
          (dA_prev)
    """
    # Set dimensions
    m, h_new, w_new, c_new = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # create empty output
    dA_prev = np.zeros(A_prev.shape)

    # fill output
    for frame in range(m):
        for x in range(w_new):
            for y in range(h_new):
                for z in range(c_new):
                    if mode == 'avg':
                        average = dA[frame, x, y, z] / kh / kw
                        dA_prev[frame,
                                x*sh:x*sh+kh,
                                y*sw:y*sw+kw,
                                z] += np.full(kernel_shape, average)
                    if mode == 'max':
                        filter = A_prev[frame,
                                        x*sh:x*sh+kh,
                                        y*sw:y*sw+kw,
                                        z]
                        mask = (filter == np.max(filter))
                        dA_prev[frame,
                                x*sh:x*sh+kh,
                                y*sw:y*sw+kw,
                                z] += mask * dA[frame, x, y, z]
    return dA_prev
