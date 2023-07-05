#!/usr/bin/env python3
"""Task 1"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Discriminator(nn.Module):
    """
    defines the generator: class Discriminator(nn.Module):
    Define the __init__ construct with these parameters: (self, input_size,
    hidden_size, output_size)
    Make sure you define the feed-forward function inside of the class def
    forward(self, x):
    The network should have three layers and three sigmoid activation
    functions after each layer.
    The layers and activation functions should be contained inside of a
    nn.Sequential class.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Defines variables for discriminator instance
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Feed-forward function"""
        return self.main(x)
