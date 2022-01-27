import numpy as np
import numpy.random as random
import matplotlib.pyplot as plt
from common_function import sigmoid

from typing import List


class Neuron:
    def __init__(self, weights: np.ndarray):
        self.weights = weights
        self.output = None
        self.output_derivative = None

    def calc_output(self, inputs):
        self.output = sigmoid(self.linear_feedforward(inputs))
        self.output_derivative = sigmoid(self.output, derivative=True)
        return self.output

    def linear_feedforward(self, inputs):
        return np.dot(self.weights, inputs).item()

    def correct_weights(self, input_layer: List, expected, l_rate):
        error = self.output - expected
        complex_derivative_error = error * self.output_derivative
        for idx, input_neuron in enumerate(input_layer):
            correction = complex_derivative_error * input_neuron.output
            new_weight = self.weights.take(idx) - l_rate * correction * input_neuron.output
            self.weights.put(idx, new_weight, mode='raise')


class NeuralNetwork:
    def __init__(self, neurons_in_layers: []):
        self.layers = []
        layers_amount = len(neurons_in_layers)

        for i in range(1, layers_amount):
            layer = []

            neurons_in_layer = neurons_in_layers[i]
            if i != layers_amount - 1:
                neurons_in_layer = neurons_in_layer + 1

            neurons_weights_size = neurons_in_layers[i - 1]
            if i != 1:
                neurons_weights_size = neurons_weights_size + 1

            for j in range(neurons_in_layer):
                layer.append(Neuron(weights=random.rand(1, neurons_weights_size)))

            self.layers.append(layer)

    def feedforward(self, input_vector: []):
        net_output = []
        layer_output = []
        for layer in self.layers:
            if layer == self.layers[0]:
                layer_output = input_vector

            for neuron in layer:
                neuron_out = neuron.calc_output(layer_output)
                net_output.append(neuron_out)

            if layer == self.layers[-1]:
                return net_output
            else:
                layer_output.clear()
                layer_output.extend(net_output)
                net_output.clear()
        return net_output

    def backpropagation(self, input, expected, l_rate):
        layers_amount = len(self.layers)
        for i in reversed(range(layers_amount)):
            current_layer = self.layers[i]
            if i == layers_amount:
                prev_layer = input
            else:
                prev_layer = self.layers[i - 1]
            for idx, neuron in enumerate(current_layer):
                neuron.correct_weights(input_layer=prev_layer, expected=expected, l_rate=l_rate)

    def train(self, train_set: np.ndarray, expected_output_set: np.ndarray, epochs: int, l_rate: float):
        error = 0.0
        prev_error = 0.0
        scaled_rate: float = l_rate
        for epoch in range(epochs):
            # TODO: fix regularization logic
            if prev_error != 0 and np.abs(error - np.abs(prev_error)) <= 5:
                scaled_rate = scaled_rate * 5
            else:
                scaled_rate = l_rate
            prev_error = np.copy(error)
            error = 0.0

            for i in range(train_set.size):
                input_vector = train_set.take(i)
                expected = expected_output_set.take(i)
                out = self.feedforward([input_vector])
                self.backpropagation(input=input_vector, expected=expected, l_rate=scaled_rate)
                error += (expected - out) ** 2
            print('>epoch=%d, lrate=%.3f, error=%.10f' % (epoch, scaled_rate, error))


# cubic equation prediction

nn = NeuralNetwork([1, 1000, 10, 1])

points_sample = 100
X = 2 * np.random.uniform(-1, 1, points_sample)
y = np.power(X, 3) + 7

nn.train(X, y, epochs=100, l_rate=0.003)

test_data_size = 1000
noise = np.random.normal(-0.2, 0.2, points_sample)

test_data = X + noise
result = np.empty(0)
for val in test_data.flatten():
    result = np.append(result, nn.feedforward([val]))

plt.scatter(X.flatten(), y.flatten(), color='blue')
plt.scatter(test_data.flatten(), result.flatten(), color='red')
plt.show()

#  SINUS PREDICTION (FAILING)

# nn = NeuralNetwork([1, 250, 1])
#
# points_sample = 100
# X = 2 * np.pi * np.random.rand(points_sample).reshape(1, -1)
# y = np.sin(X)
#
# nn.train(X, y, epochs=500, l_rate=100.0)
#
# test_data_size = 1000
# noise = np.random.normal(-0.2, 0.2, points_sample)
#
# test_data = X + noise
# result = np.empty(0)
# for val in test_data.flatten():
#     result = np.append(result, nn.feedforward([val]))
#
# plt.scatter(X.flatten(), y.flatten(), color='blue')
# plt.scatter(test_data.flatten(), result.flatten(), color='red')
# plt.show()
