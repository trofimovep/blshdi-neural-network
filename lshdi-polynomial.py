"""
    Example of using :py:class:`lshdi.LSHDI`
"""

import numpy as np
from lshdi import LSHDI
from matplotlib import pyplot as plt

nn = LSHDI(1, 50, 1)

# generate training data
points_sample = 30000
X = 2 * np.random.uniform(-1, 1, points_sample)
y = np.power(X, 4) + np.power(X, 2) + 7

# train network
nn.train(X, y)

# generate test data
test_data = 2 * np.random.uniform(-1, 1, points_sample)

# check the result
result = np.empty(0)
for val in test_data.flatten():
    result = np.append(result, nn.feedforward(val))

plt.scatter(X.flatten(), y.flatten(), color='blue')
plt.scatter(test_data.flatten(), result.flatten(), color='red')
plt.show()
