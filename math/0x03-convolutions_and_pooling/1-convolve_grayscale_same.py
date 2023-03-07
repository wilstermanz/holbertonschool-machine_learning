#!/usr/bin/env python3
"""Task 1"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    performs a same convolution on grayscale images:

        images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
          the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        if necessary, the image should be padded with 0’s
        You are only allowed to use two for loops; any other loops of any kind
          are not allowed
        Returns: a numpy.ndarray containing the convolved images
        """
    # input dimensions
    m, h, w = images.shape
    kh, kw = kernel.shape

    # calculate padding
    pad_along_height = max((h - 1) + kh - h, 0)
    pad_along_width = max((w - 1) + kw - w, 0)
    pad_top = pad_along_height // 2
    pad_bottom = pad_along_height - pad_top
    pad_left = pad_along_width // 2
    pad_right = pad_along_width - pad_left

    # pad images
    images_padded = np.pad(images, ((0, 0),
                                    (pad_top, pad_bottom),
                                    (pad_left, pad_right)))

    # make output
    output = np.zeros((m, h, w))

    # loop over output
    for x in range(h):
        for y in range(w):
            output[:, x, y] = np.sum(
                kernel * images_padded[:, x:x+kh, y:y+kw], axis=(1, 2))

    return output
