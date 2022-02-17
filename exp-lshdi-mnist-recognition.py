import numpy as np
from lshdi import LSHDI
from matplotlib import pyplot as plt
from tensorflow.keras.datasets import mnist

# load test data
(trainX, train_y), (testX, testy) = mnist.load_data()

import time

start_time = time.time()

# define a train size (for reducing calculation time)
train_size = 1000
# siz = trainX.size
trainX = trainX[0:train_size]
train_y = train_y[0:train_size]

# convert the image matrix to vector
trainX_vectors = [image_matrix.ravel() for image_matrix in trainX]

# form the train output: it is an array (length 10) which consist of {0 1}
# and the index of 1 defines the number corresponding to the image
trainy = []
for y in train_y:
    val = np.zeros(10)
    val.put(y, 1)
    trainy.append(val)

# define the dependency from hidden neurons amount
sizes = []
accuracy = []
hidden_neurons_amount = 980
while hidden_neurons_amount < 28 * 28 * 10 + 1:
    print(hidden_neurons_amount)
    sizes.append(hidden_neurons_amount)
    nn = LSHDI(28 * 28, hidden_neurons_amount, 10)
    nn.train(trainX_vectors, trainy)
    # test it
    success: int = 0
    for i in range(testy.size):
        res = nn.feedforward(testX[i].ravel())
        answer = np.where(res == max(res))
        if answer == testy[i]:
            success += 1
    percent = success / testy.size * 100
    accuracy.append(percent)
    hidden_neurons_amount = hidden_neurons_amount + 56

plt.scatter(sizes, accuracy, color='blue', marker='v')
plt.show()
