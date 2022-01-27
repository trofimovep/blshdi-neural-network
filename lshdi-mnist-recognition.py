# docker pull tensorflow/tensorflow:latest
# docker run -it -p 8888:8888 tensorflow/tensorflow:latest-jupyter
import numpy as np
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from LSHDI2 import LSHDI

(trainX, train_y), (testX, testy) = mnist.load_data()

siz = 6000
trainX = trainX[0:siz]
train_y = train_y[0:siz]

trainy = []
for y in train_y:
    val = np.zeros(10)
    val.put(y, 1)
    trainy.append(val)

# testX = testX[0:100]
# testy = testy[0:100]

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(trainX[i], cmap=plt.get_cmap('gray'))
plt.show()

trainX_vectors = [image_matrix.ravel() for image_matrix in trainX]

nn = LSHDI(28 * 28, 120 * 10, 10)
nn.train(trainX_vectors, trainy)

success: int = 0
for i in range(testy.size):
    res = nn.feedforward(testX[i].ravel())
    answer = np.where(res == max(res))
    if answer == testy[i]:
        success += 1

percent = success / testy.size * 100
print('Result: ')
print(percent)
