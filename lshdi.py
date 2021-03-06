"""
    LSHDI - linear solution to higher dimensional interlayer networks

    This net was written by the following article: https://arxiv.org/ftp/arxiv/papers/1207/1207.3368.pdf
    It consists of input, one hidden layer, one output linear layer.
    A net training is provided by matrix pseudoinverse.
"""
import numpy as np


def sigmoid(x, derivative=False):
    if derivative:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))


class LSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        # each row in a layer (matrix) is the neuron weights
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
        train_hidden_out = np.column_stack(self.calc_hidden_output(train) for train in train_set)
        # print('Start calculating pinv...')
        pinv_train_hidden_out = np.linalg.pinv(np.transpose(train_hidden_out))
        # print('Start calculating a new weights...')
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)
