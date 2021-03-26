#!/usr/bin/env python3
import numpy as np
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
