"""
    lshdi using cline analogue formula
    @link https://www.elibrary.ru/item.asp?id=42804101
"""

import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class LSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        # each row in a layer (matrix) is a neuron weights
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

        # TODO: разбить попарно на блоки и посчитать по формуле

        train_hidden_out = np.column_stack(self.calc_hidden_output(train) for train in train_set)
        print('Start calculating pinv...')
        pinv_train_hidden_out = np.linalg.pinv(np.transpose(train_hidden_out))
        print('Start calculating a new weights...')
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)
