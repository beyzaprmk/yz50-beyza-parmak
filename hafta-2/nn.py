from engine import Value
import random


class Neuron:
    def __init__(self, nin):
        self.weights = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.bias = Value(0.0)

    def __call__(self, x):
        total = self.bias

        for wi, xi in zip(self.weights, x):
            #z=w1​x1​+w2​x2​+⋯+wn​xn​+b
            total = total + wi * xi

        return total.tanh()

    def parameters(self):
        return self.weights + [self.bias]




class Layer:
    def __init__(self, nin, nout):
        self.neurons = [Neuron(nin) for _ in range(nout)]

    def __call__(self, x):
        outputs = [neuron(x) for neuron in self.neurons]

        if len(outputs) == 1:
            return outputs[0]


        return outputs

    def parameters(self):
        params = []

        for neuron in self.neurons:
            params.extend(neuron.parameters())


        return params



class MLP:
    def __init__(self, nin, nouts):
        sizes = [nin] + nouts

        self.layers = [
            Layer(sizes[i], sizes[i + 1])
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)

        return x

    def parameters(self):
        params = []

        for layer in self.layers:
            params.extend(layer.parameters())

        return params