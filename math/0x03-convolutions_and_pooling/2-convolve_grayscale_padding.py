#!/usr/bin/env python3
"""Task 2"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    performs a convolution on grayscale images with custom padding:

        images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
          the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        padding is a tuple of (ph, pw)
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0s
        You are only allowed to use two for loops; any other loops of any kind
          are not allowed
        Returns: a numpy.ndarray containing the convolved images
        """
    # input dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape
    ph, pw = padding

    # make output
    output = np.zeros((m, h + (2 * ph) - kh + 1, w + (2 * pw) - kw + 1))

    # pad images
    images_padded = np.pad(images, ((0, 0),
                                    (ph, ph),
                                    (pw, pw)))

    # loop over output
    for x in range(output.shape[2]):
        for y in range(output.shape[1]):
            output[:, y, x] = np.sum(
                kernel * images_padded[:, y:y+kh, x:x+kw], axis=(1, 2))

    return output
