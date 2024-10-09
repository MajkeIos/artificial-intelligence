import numpy as np
import sys


def read_input(file_to_read):
    my_file = open(file_to_read, "r")
    res = []
    for line in my_file.read().split('\n'):
        if len(line) < 100:
            continue
        line = [float(x) for x in line.split(',')]
        res.append((line[0], np.array(line[1:]).reshape(1, len(line[1:]))))
    return res


class Layer:
    def __init__(self, in_size, out_size, last_layer):
        self.in_size = in_size
        self.out_size = out_size
        self.last_layer = last_layer
        self.W = np.random.normal(0.0, 0.4, (in_size, out_size))
        self.B = np.random.normal(0.0, 0.4, (1, out_size))
        self.Z = None
        self.A_prev = None

    def forward_prop(self, A_prev):
        self.A_prev = A_prev
        self.Z = np.matmul(A_prev, self.W)
        self.Z = self.Z + self.B
        if not self.last_layer:
            A = np.maximum(0, self.Z)
            return A
        return self.Z

    def backward_prop(self, dA):
        dZ = dA
        if not self.last_layer:
            dZ = (self.Z > 0) * dA
        dW = np.matmul(self.A_prev.T, dZ)
        dB = dZ
        dA_prev = np.matmul(dZ, self.W.T)
        self.W -= learning_rate * dW
        self.B -= learning_rate * dB
        return dA_prev


learning_rate = 0.03
training_file = sys.argv[1]
testing_file = sys.argv[2]
training_data = read_input(training_file)
training_data = training_data[2000:]
testing_data = read_input(testing_file)
epoch = 2
layers = [Layer(784, 128, False), Layer(128, 10, False)]
for i in range(epoch):
    for test_case in training_data:
        expected_output = test_case[0]
        training_input = test_case[1] / 255.0
        for layer in layers:
            training_input = layer.forward_prop(training_input)
        softmax = np.exp(training_input - np.max(training_input))
        softmax /= softmax.sum()
        softmax[0][int(expected_output)] -= 1.0
        for layer in reversed(layers):
            softmax = layer.backward_prop(softmax)

for test_case in testing_data:
    expected_output = test_case[0]
    testing_input = test_case[1] / 255.0
    for layer in layers:
        testing_input = layer.forward_prop(testing_input)
    result = np.argmax(testing_input)
    print(result)
