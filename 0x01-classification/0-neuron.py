#!/usr/bin/env python3
import numpy as np
"""Neuron class"""
class Neuron:
"""Class that defines a single neuron """
    def __init__(self, nx):
"""Class constructor"""
        if type(nx) is not int:
            raise TypeError('nx must be an integer')
        if (nx<1):
            raise ValueError('nx must be a positive integer')
        self.W = np.random.normal()
        self.b = 0
        self.A = 0
