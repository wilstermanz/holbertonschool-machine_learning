#!/usr/bin/env python3
"""Task 0"""
import numpy as np

def convolve_grayscale_valid(images, kernel): 
    """
    performs a valid convolution on grayscale images:

        images is a numpy.ndarray with shape (m, h, w) containing multiple
          grayscale images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
        kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for
          the convolution
            kh is the height of the kernel
            kw is the width of the kernel
        You are only allowed to use two for loops; any other loops of any kind
          are not allowed
        Returns: a numpy.ndarray containing the convolved images
    """
    # input dimensions
    m, input_h, input_w = images.shape[0], images.shape[1], images.shape[2]
    kernel_w, kernel_h = kernel.shape[0], kernel.shape[1]

    # output dimensions
    output_h = input_h - kernel_h + 1
    output_w = input_w - kernel_w + 1

    # convolution output
    output = np.zeros((m, output_h, output_w))

    # loop over every pixel of the output
    for x in range(output_w):
        for y in range(output_h):
            output[:, x, y] = np.sum(
                kernel * images[:, x:x+kernel_h, y:y+kernel_w], axis=(1, 2))

    return output
