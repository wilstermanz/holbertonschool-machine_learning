#!/usr/bin/env python3
"""Task 0"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class Generator(nn.Module):
    """
    subclass that defines the generator: class Generator(nn.Module):
    Define the __init__ construct with these parameters: (self, input_size, hidden_size, output_size)
    Make sure you define the feed-forward function inside of the class def forward(self, x):
    The network should have three layers and two tanh activation functions after the first and second layer.
    The layers and activation functions should be contained inside of a nn.Sequential wrapper class.
    """
    def __init__(self, input_size, hidden_size, output_size):
        """
        Defines variables for generator instance
        """
        super().__init__()
        self.main = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        """Performs forward feed"""
        return self.main(x)
