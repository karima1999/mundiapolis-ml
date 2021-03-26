#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
     def __init__(self, nx):
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    @staticmethod
    def plot_training_cost(list_iterations, list_cost, graph):

        if graph:
            plt.plot(list_iterations, list_cost)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training cost')
            plt.show()

    @staticmethod
    def print_verbose_for_step(iteration, cost, verbose, step, list_cost):
        if verbose and iteration % step == 0:
            print('Cost after ' + str(iteration) + ' iterations: ' + str(cost))
        list_cost.append(cost)

    def forward_prop(self, X):

        A_prev = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-A_prev))
        return self.__A

    def cost(self, Y, A):
        m = Y.shape[1]
        cost = - (1 / m) * np.sum(np.multiply(Y, np.log(A)) +
                                  np.multiply(1 - Y, np.log(1.0000001 - A)))
        return cost

    def evaluate(self, X, Y):
        self.forward_prop(X)
        return np.where(self.A <= 0.5, 0, 1), self.cost(Y, self.A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        m = Y.shape[1]
        d_ay = A - Y
        gradient = np.matmul(d_ay, X.T) / m
        db = np.sum(d_ay) / m
        self.__W -= gradient * alpha
        self.__b -= db * alpha

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        
        if type(iterations) is not int:
            raise TypeError('iterations must be an integer')
        if iterations < 0:
            raise ValueError('iterations must be a positive integer')
        if type(alpha) is not float:
            raise TypeError('alpha must be a float')
        if alpha < 0:
            raise ValueError('alpha must be positive')

        if verbose and graph:
            if type(step) is not int:
                raise TypeError('step must be an integer')
            if not 0 <= step <= iterations:
                raise ValueError('step must be positive and <= iterations')

        list_cost = list()
        list_iterations = [*list(range(iterations)), iterations]

        for i in list_iterations:
            A, cost = self.evaluate(X, Y)
            self.print_verbose_for_step(i, cost, verbose, step, list_cost)
            self.gradient_descent(X, Y, self.A, alpha)

        self.plot_training_cost(list_iterations, list_cost, graph)
        return A, cost
