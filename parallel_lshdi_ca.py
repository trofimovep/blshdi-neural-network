"""
    lshdi using cline analogue formula
    @link https://www.elibrary.ru/item.asp?id=42804101
"""
from threading import Thread

import numpy as np
from utils import divide_on_4
from cline_analogue import two_block_inverse_with_pinv, two_block_inverse_and_put, calc_pinv_by_rows_and_inversed, \
    two_block_inverse_then_calc_and_put
from tensorflow.keras.datasets import mnist


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class PLSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        # each row in a layer (matrix) is neuron weights
        self.output = None
        self.hidden_output = None
        self.hidden_layer = 2 * np.random.rand(hidden_neurons_amount, input_vector_size) - 1
        self.output_layer = 2 * np.random.rand(output_vector_size, hidden_neurons_amount) - 1

    def calc_hidden_output(self, input_vec):
        if np.isscalar(input_vec):
            self.hidden_output = sigmoid(self.hidden_layer * input_vec)
        else:
            self.hidden_output = sigmoid(np.matmul(self.hidden_layer, input_vec))
        return self.hidden_output

    def feedforward(self, input_vec: np.ndarray):
        self.hidden_output = self.calc_hidden_output(input_vec=input_vec)
        self.output = np.matmul(np.transpose(self.output_layer), self.hidden_output)
        return self.output

    def train(self, train_set: np.ndarray, train_out_set: np.ndarray):
        train_hidden_out = np.transpose(
            np.column_stack(self.calc_hidden_output(train) for train in train_set)
        )
        # print('Start calculating pinv...')
        I, II, III, IV = divide_on_4(train_hidden_out.shape[1])
        iblocks = {}
        t12 = Thread(
            target=two_block_inverse_then_calc_and_put,
            args=(iblocks, '1-2', train_hidden_out[:, :I + 1], train_hidden_out[:, I + 1:II + 1], train_hidden_out[:, II + 1:])
        )
        t12.start()

        t34 = Thread(
            target=two_block_inverse_then_calc_and_put,
            args=(iblocks, '3-4', train_hidden_out[:, III:IV + 1], train_hidden_out[:, IV + 1:], train_hidden_out[:, :II + 1])
        )
        t34.start()

        t12.join()
        t34.join()

        # pinv_train_hidden_out = two_block_inverse_with_pinv(
        #     train_hidden_out[:, :II + 1], iblocks['1-2'],
        #     train_hidden_out[:, II + 1:], iblocks['3-4']
        # )

        pinv_train_hidden_out = calc_pinv_by_rows_and_inversed(
            iblocks['row1-2'], iblocks['row3-4'], iblocks['1-2'], iblocks['3-4']
        )

        # print('Start calculating a new weights...')
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)

# test
# load test data
# (trainX, train_y), (testX, testy) = mnist.load_data()
#
# import time
#
# start_time = time.time()
#
# # define a train size (for reducing calculation time)
# train_size = 600
# # siz = trainX.size
# trainX = trainX[0:train_size]
# train_y = train_y[0:train_size]
#
# # convert the image matrix to vector
# trainX_vectors = [image_matrix.ravel() for image_matrix in trainX]
#
# # form the train output: it is an array (length 10) which consist of {0 1}
# # and the index of 1 defines the number corresponding to the image
# trainy = []
# for y in train_y:
#     val = np.zeros(10)
#     val.put(y, 1)
#     trainy.append(val)
#
# # define neural net and train it
# nn = PLSHDI(28 * 28, 28 * 10, 10)
# nn.train(trainX_vectors, trainy)
#
# # test it
# success: int = 0
# for i in range(testy.size):
#     res = nn.feedforward(testX[i].ravel())
#     answer = np.where(res == max(res))
#     if answer == testy[i]:
#         success += 1
#
# percent = success / testy.size * 100
# print('Success recognized images: ', percent, '%')
#
# print("--- %s seconds ---" % (time.time() - start_time))
