"""
    lshdi using cline analogue formula
    @link https://www.elibrary.ru/item.asp?id=42804101
"""

import numpy as np
from utils import divide_on_4
from cline_analogue import two_block_inverse, two_block_inverse_with_pinv
from tensorflow.keras.datasets import mnist


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class LSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        # each row in a layer (matrix) is a neuron weights
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
        # tho = np.transpose(train_hidden_out)
        print('Start calculating pinv...')
        I, II, III, IV = divide_on_4(train_hidden_out.shape[1])
        first_block = train_hidden_out[:, :I+1]
        second_block = train_hidden_out[:, I+1:II+1]

        third_block = train_hidden_out[:, III:IV+1]
        fourth_block = train_hidden_out[:, IV+1:]


        # распараллелить
        pinv_I_II = two_block_inverse(first_block, second_block)
        pinv_III_IV = two_block_inverse(third_block, fourth_block)
        # TODO: если известны псевдообратные

        pinv_train_hidden_out = two_block_inverse_with_pinv(
            train_hidden_out[:, :II+1], pinv_I_II,
            train_hidden_out[:, II+1:], pinv_III_IV
        )

        print('Start calculating a new weights...')
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)


# test
# load test data
(trainX, train_y), (testX, testy) = mnist.load_data()

import time
start_time = time.time()

# define a train size (for reducing calculation time)
train_size = 6000
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

# define neural net and train it
nn = LSHDI(28 * 28, 28 * 28 * 10, 10)
nn.train(trainX_vectors, trainy)

# test it
success: int = 0
for i in range(testy.size):
    res = nn.feedforward(testX[i].ravel())
    answer = np.where(res == max(res))
    if answer == testy[i]:
        success += 1

percent = success / testy.size * 100
print('Success recognized images: ', percent, '%')

print("--- %s seconds ---" % (time.time() - start_time))
