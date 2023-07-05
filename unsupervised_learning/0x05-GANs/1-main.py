#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

disTester = __import__('1-discriminator').Discriminator

D_Test = disTester(1,1,1)

print(D_Test)
