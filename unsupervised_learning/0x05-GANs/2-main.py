#!/usr/bin/env python3

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

samplerTest = __import__('2-sample_Z').sample_Z

print(samplerTest(0,1,"D"))

print(samplerTest(0,1,"G"))

print(samplerTest(0,1,"F"))
