#!/usr/bin/env python3
"""Task 3"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    performs a convolution on grayscale images:

        images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
          the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
                ph is the padding for the height of the image
                pw is the padding for the width of the image
            the image should be padded with 0’s
        stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image
        You are only allowed to use two for loops; any other loops of any kind
          are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    # input dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    # calculate padding
    if type(padding) is tuple:
        ph, pw = padding
    if padding == 'valid':
        ph, pw = 0, 0
    if padding == 'same':
        ph = ((h - 1) * sh + kh - h) // 2 + 1
        pw = ((w - 1) * sw + kw - w) // 2 + 1

    # pad images
    images_padded = np.pad(images, ((0, 0),
                                    (ph, ph),
                                    (pw, pw)))

    # make output
    output = np.zeros((m,
                      (h + (2 * ph) - kh) // sh + 1,
                      (w + (2 * pw) - kw) // sw + 1))

    # loop over output
    for x in range(output.shape[2]):
        for y in range(output.shape[1]):
            output[:, y, x] = np.sum(
                kernel * images_padded[:,
                                       y*sh:y*sh+kh,
                                       x*sw:x*sw+kw],
                axis=(1, 2))
    return output
