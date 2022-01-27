"""
    LSHDI
    (linear solution to higher dimensional interlayer networks
    https://arxiv.org/ftp/arxiv/papers/1207/1207.3368.pdf

    * one layer net
    Layer consist of input, one hidden layer, one output linear layer

    Advantages:
    very	quick	to	compute, and	avoids	the	problems	of	stability	and convergence	on	local	minima
"""
import matplotlib.pyplot as plt
import numpy as np

from common_function import sigmoid as sig


class LSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        # each row in a layer (matrix) is a neuron weights
        self.hidden_layer = 2 * np.random.rand(hidden_neurons_amount, input_vector_size) - 1
        self.output_layer = 2 * np.random.rand(output_vector_size, hidden_neurons_amount) - 1

    def calc_hidden_output(self, input_vec):
        if np.isscalar(input_vec):
            self.hidden_output = sig(self.hidden_layer * input_vec)
        else:
            self.hidden_output = sig(np.matmul(self.hidden_layer, input_vec))
        return self.hidden_output

    def feedforward(self, input_vec: np.ndarray):
        self.hidden_output = self.calc_hidden_output(input_vec=input_vec)
        self.output = np.matmul(np.transpose(self.output_layer), self.hidden_output)
        return self.output

    def train(self, train_set: np.ndarray, train_out_set: np.ndarray):
        # train_hidden_out = [self.calc_hidden_output(train) for train in train_set]
        train_hidden_out = np.column_stack(self.calc_hidden_output(train) for train in train_set)
        print('Start calculating pinv...')
        pinv_train_hidden_out = np.linalg.pinv(np.transpose(train_hidden_out))
        print('Start calculating a new weights...')
        self.output_layer = np.matmul(pinv_train_hidden_out, train_out_set)


points_sample = 200

nn = LSHDI(1, 9, 1)
X = 2 * np.random.uniform(-1, 1, points_sample)
y = np.power(X, 3) + np.power(X, 2) + 7
# y = np.sin(X)

nn.train(X, y)

# noise = np.random.normal(-0.2, 0.2, points_sample)
#
# test_data = X + noise
test_data = 2 * np.random.uniform(-1, 1, points_sample)

result = np.empty(0)
for val in test_data.flatten():
    result = np.append(result, nn.feedforward(val))

plt.scatter(X.flatten(), y.flatten(), color='blue')
plt.scatter(test_data.flatten(), result.flatten(), color='red')
plt.show()
