#!/usr/bin/env python3


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
genTester = __import__('0-generator').Generator

G_Test = genTester(1,1,1)

print(G_Test)
