#!/usr/bin/env python3
"""Task 2"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def sample_Z(mu, sigma, sampleType, size=(1, 1)):
    """
    creates input for the generator and discriminator:

    mu Should be the mean of the distribution
    sigma Should be the standard deviation of the distribution
    sampleType Should be a variable that selects which model to sample for.
        The variable should accept a "G" or "D" as string values.
    The input data for discrimintator should be from a normal distribution (it
    will also need random sampling in the training phase).
    The input data for generator should be random sampling.
    The function should return a torch.Tensor type for both generator and
    discriminator if the parameters are correct.
        It should return 0 otherwise.
    """
    if sampleType == 'G':
        return torch.normal(mu, sigma, size)

    if sampleType == 'D':
        return torch.randn(size)

    return 0