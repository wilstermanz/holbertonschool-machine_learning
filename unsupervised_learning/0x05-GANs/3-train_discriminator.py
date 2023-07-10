#!/usr/bin/env python3
"""Task 3"""
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


def train_dis(Gen, Dis, dInputSize, gInputSize,
              mbatchSize, steps, optimizer, crit):
    """
    Gen, and Dis are the Discriminator and Generator Objects.
    dInputSize is the input size of Discriminator input data.
    gInputSize is the input size of Generator input data.
    mbatchSize should be the batch size for training.
    steps should be the number of steps for training.
    optimizer should be a stochastic gradient descent optimizer object.
    The function should return the two item methods that belong to loss
    entropy class for real and fake
    The crit should be a BCEloss function.
    Should use both random noise, and normal distribution for sampling
    The 4 moments should be used in processing the sample.
    The function should return the error estimate of the fake and real data,
    along with the fake and real data sets of type torch.Tensor().
    """
    for _ in range(steps):
        # Create real data
        realData = sample_Z(0., 1., 'D', (mbatchSize, dInputSize))
        realLabels = torch.ones((mbatchSize, 1))

        # Create fake data
        fakeData = Gen(sample_Z(0., 1., 'G', (mbatchSize, gInputSize)))
        fakeLabels = torch.zeros((mbatchSize, 1))

        # Combine data
        allData = torch.cat((realData, fakeData))
        allLabels = torch.cat((realLabels, fakeLabels))

        # Train discriminator
        Dis.zero_grad()
        output = Dis(allData)
        loss = crit(output, allLabels)
        loss.backward()
        optimizer.step()

    
    return 
