"""
    LSHDI
    (linear solution to higher dimensional interlayer networks
    https://arxiv.org/ftp/arxiv/papers/1207/1207.3368.pdf

    * one layer net
    Layer consist of input, one hidden layer, one output linear layer
"""
import numpy as np
from common_function import sigmoid


class Neuron:
    def __init__(self, weights_size: int, is_output: bool = False):
        self.is_output = is_output
        self.weights = np.random.rand(weights_size)

    def feedforward(self, input_vector: np.ndarray):
        self.linear_out = np.dot(self.weights, input_vector)
        if self.is_output is True:
            return self.linear_out
        else:
            self.out = sigmoid(self.linear_out)
            return self.out


class LSHDI:
    def __init__(self, input_vector_size: int, hidden_neurons_amount: int, output_vector_size: int):
        self.hidden_layer = []
        for i in range(hidden_neurons_amount):
            self.hidden_layer.append(Neuron(weights_size=input_vector_size))
        self.output_layer = []
        for i in range(output_vector_size):
            self.output_layer.append(Neuron(weights_size=hidden_neurons_amount, is_output=True))

    def feedforward(self, input_vector: np.ndarray):
        hidden_output = [neuron.feedforward(input_vector=input_vector) for neuron in self.hidden_layer]
        out = [out_neuron.feedforward(input_vector=hidden_output) for out_neuron in self.output_layer]
        return out

    def find_linear_layer_weights(self, hidden_outputs, outputs):
        pinv = np.linalg.pinv(hidden_outputs)
        new_weights = pinv.dot(outputs)
        return new_weights

    def train(self, train_input_data: list, train_output_data: list):
        # net_outputs = []
        hidden_outputs = []
        for idx, train in enumerate(train_input_data):
            self.feedforward(train)
            # net_outputs.append(out)
            for neuron in self.hidden_layer:
                hidden_outputs.append(neuron.out)
        new_weights = self.find_linear_layer_weights(hidden_outputs, train_output_data[idx])
        # /TODO ПРАВКА ВЕСОВ
        for idx, neuron in enumerate(self.output_layer):
            neuron.weights = new_weights[idx, idx]
        print(new_weights)



# Test array
# nn = LSHDI(2, 2, 2)
# input_vec = np.random.rand(2)
# out = nn.feedforward(input_vector=input_vec)
# print(out)


# Test data

points_sample = 100

nn = LSHDI(1, 2, 1)
X = 2 * np.random.uniform(-1, 1, points_sample)
y = np.power(X, 3) + 7

nn.train(X.tolist(), y.tolist())

noise = np.random.normal(-0.2, 0.2, points_sample)

test_data = X + noise
result = np.empty(0)
for val in test_data.flatten():
    result = np.append(result, nn.feedforward([val]))

plt.scatter(X.flatten(), y.flatten(), color='blue')
plt.scatter(test_data.flatten(), result.flatten(), color='red')
plt.show()
