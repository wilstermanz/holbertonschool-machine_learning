#!/usr/bin/env python3
"""Tasks 16 - 23 for Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initializes an instance of DeepNeuralNetwork"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], prev) * np.sqrt(2 / prev)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """"cache getter"""
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""

        layers, cache, weights = self.__L, self.__cache, self.__weights
        cache['A0'] = X

        for i in range(1, layers + 1):
            z = np.matmul(weights['W{}'.format(i)], cache['A{}'.format(i - 1)]
                          ) + weights['b{}'.format(i)]
            cache['A{}'.format(i)] = 1 / (1 + np.exp(-z))

        return cache['A{}'.format(layers)], cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)[1]
        return np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        prediction = np.where(self.forward_prop(X)[0] >= .5, 1, 0)
        return prediction, self.cost(Y, self.__cache["A{}".format(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Evaluates the neural network’s predictions"""
#!/usr/bin/env python3
"""Tasks 16 - 23 for Deep Neural Network"""
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""

    def __init__(self, nx, layers):
        """Initializes an instance of DeepNeuralNetwork"""

        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        prev = nx
        for i in range(len(layers)):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            self.__weights["W{}".format(i + 1)] = np.random.randn(
                layers[i], prev) * np.sqrt(2 / prev)
            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
            prev = layers[i]

    @property
    def L(self):
        """L getter"""
        return self.__L

    @property
    def cache(self):
        """"cache getter"""
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""

        layers, cache, weights = self.__L, self.__cache, self.__weights
        cache['A0'] = X

        for i in range(1, layers + 1):
            z = np.matmul(weights['W{}'.format(i)], cache['A{}'.format(i - 1)]
                          ) + weights['b{}'.format(i)]
            cache['A{}'.format(i)] = 1 / (1 + np.exp(-z))

        return cache['A{}'.format(layers)], cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = np.shape(Y)[1]
        return np.sum((-Y * np.log(A)) - ((1 - Y) * np.log(1.0000001 - A))) / m

    def evaluate(self, X, Y):
        """Evaluates the neural network’s predictions"""
        prediction = np.where(self.forward_prop(X)[0] >= .5, 1, 0)
        return prediction, self.cost(Y, self.__cache["A{}".format(self.__L)])

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Evaluates the neural network’s predictions"""
        input_layer, output_layer, weights = 1, self.__L, self.__weights
        m = np.shape(Y)[1]
        for layer in range(output_layer, 0, -1):
            if layer == output_layer:
                dz = cache["A{}".format(layer)] - Y
            if layer < output_layer:
                a = cache["A{}".format(layer)]
                dz = da * a * (1 - a)
            dW = np.matmul(dz, cache["A{}".format(layer - 1)].T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            da = np.matmul(weights["W{}".format(layer)].T, dz)
            weights["W{}".format(layer)] -= alpha * dW
            weights["b{}".format(layer)] -= alpha * db
