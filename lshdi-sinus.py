"""
    Example of using :py:class:`lshdi.LSHDI`
"""

import numpy as np
from lshdi import LSHDI
from matplotlib import pyplot as plt

nn = LSHDI(1, 3, 1)

# generate training data
points_sample = 20
X = 2 * np.random.uniform(-1, 1, points_sample)
y = np.sin(X) + 5

# train network
nn.train(X, y)

# generate test data
test_data = 2 * np.random.uniform(-1, 1, points_sample + 20)

# check the result
result = np.empty(0)
for val in test_data.flatten():
    result = np.append(result, nn.feedforward(val))


X_sin = 2 * np.random.uniform(-1, 1, 1000)
y_sin = np.sin(X_sin) + 5

plt.scatter(X_sin.flatten(), y_sin.flatten(), color='gray', marker=".")
plt.scatter(test_data.flatten(), result.flatten(), color='blue', marker='v')
plt.show()
