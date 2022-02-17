"""
    Checking the dependency accuracy on test data size
"""

import numpy as np
from lshdi import LSHDI
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

# load test data
(trainX, train_y), (testX, testy) = mnist.load_data()

import time

start_time = time.time()

sizes = []
accuracy = []

test_data_size = 300
while test_data_size < 60_000 + 1:
    sizes.append(test_data_size)

    trainX = trainX[0:test_data_size]
    train_y = train_y[0:test_data_size]
    trainX_vectors = [image_matrix.ravel() for image_matrix in trainX]
    trainy = []
    for y in train_y:
        val = np.zeros(10)
        val.put(y, 1)
        trainy.append(val)

    nn = LSHDI(28 * 28, 1500, 10)
    nn.train(trainX_vectors, trainy)

    success: int = 0
    for i in range(testy.size):
        res = nn.feedforward(testX[i].ravel())
        answer = np.where(res == max(res))
        if answer == testy[i]:
            success += 1

    percent = success / testy.size * 100
    accuracy.append(percent)
    test_data_size = test_data_size + 600
    print(test_data_size)

plt.plot(sizes, accuracy)
plt.show()

print("--- %s seconds ---" % (time.time() - start_time))
