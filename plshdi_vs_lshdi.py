from parallel_lshdi_ca import PLSHDI
from lshdi_ca import LSHDI
import numpy as np
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
import time


def main():
    print("Hello World!")
    (trainX, train_y), (testX, testy) = mnist.load_data()
    train_size = 60_000
    trainX = trainX[0:train_size]
    train_y = train_y[0:train_size]
    trainX_vectors = [image_matrix.ravel() for image_matrix in trainX]

    # form the train output: it is an array (length 10) which consist of {0 1}
    # and the index of 1 defines the number corresponding to the image
    trainy = []
    for y in train_y:
        val = np.zeros(10)
        val.put(y, 1)
        trainy.append(val)

    neurons_amount = []
    train_time_in_parallel = []
    train_time_in_one_thread = []

    step = 560
    hidden_neurons = 28
    while hidden_neurons < 28*20*10:
        print(hidden_neurons)
        neurons_amount.append(hidden_neurons)
        nn = LSHDI(28 * 28, hidden_neurons, 10)
        pnn = PLSHDI(28 * 28, hidden_neurons, 10)  # in parallel

        start = time.time()
        nn.train(trainX_vectors, trainy)
        finish = time.time()
        train_time_in_one_thread.append(finish - start)

        pstart = time.time()
        pnn.train(trainX_vectors, trainy)
        pfinish = time.time()
        train_time_in_parallel.append(pfinish - pstart)

        hidden_neurons += step

    plt.plot(neurons_amount, train_time_in_parallel, "-b", label="parallel")
    plt.plot(neurons_amount, train_time_in_one_thread, "-r", label="one thread")
    plt.legend(loc="upper left")
    plt.show()

if __name__ == "__main__":
    main()

# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(-0.75,1,100)
# y0 = np.exp(2 + 3*x - 7*x**3)
# y1 = 7-4*np.sin(4*x)
# plt.plot(x,y0,x,y1)
# plt.gca().legend(('y0','y1'))
# plt.show()
