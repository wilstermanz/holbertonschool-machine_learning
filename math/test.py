import numpy as np

np.random.seed(0)
data = np.random.poisson(5., 100).tolist()
print(data)

total = 0
for item in data:
    total += item

print(total)
print(total / 100)
