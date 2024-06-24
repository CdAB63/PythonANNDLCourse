#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:10:55 2024

@author: cdab63
"""

import random
from activation_function import ActivationFunction, SigmoidAF

class Neuron:
    
    def __init__(self, nb_of_weights, activation_function=SigmoidAF(), bias=0.0, learning_rate=0.1):
        if not isinstance(nb_of_weights, int) or nb_of_weights < 1:
            raise ValueError(f'[ERROR] invalid number of weights (>=1): {nb_of_weights}')
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError(f'[ERROR] invalid activation_function: {type(activation_function)}')
        if not isinstance(bias, (int, float)):
            raise ValueError(f'[ERROR] invalid bias: {bias}')
        if not isinstance(learning_rate, float) or learning_rate <= 0 or learning_rate >= 1.0:
            raise ValueError(f'[ERROR] invalid learning_rate: {learning_rate}')
        self.__activation_function = activation_function
        self.__bias = bias
        self.__learning_rate = learning_rate
        self.__weights = [ random.uniform(-1.0, 1.0) for _ in range(nb_of_weights) ]
        self.__output = None
        self.__delta = None
        self.__error = None
        
    def activation_function(self):
        return self.__activation_function
    
    def bias(self):
        return self.__bias
    
    def set_bias(self, x):
        if not isinstance(x, (int, float)):
            raise ValueError(f'[ERROR] invalid bias: {x}')
        self.__bias = x
       
    def learning_rate(self):
        return self.__learning_rate
    
    def set_learning_rate(self, x):
        if not isinstance(x, float) or x <= 0.0 or x >= 1.0:
            raise ValueError(f'[ERROR] invalid learning_rate {x}')
        self.__learning_rate = x
        
    def weights(self):
        return self.__weights
    
    def weight(self, idx):
        return self.__weights[idx]
    
    def nb_of_weights(self):
        if self.__weights:
            return len(self.__weights)
        else:
            return 0

    def output(self):
        return self.__output
    
    def error(self):
        return self.__error
    
    def delta(self):
        return self.__delta
    
    def feed(self, inputs):
        self.__output = self.__activation_function.fx(sum([i * w for i, w in zip(inputs, self.__weights)])+ self.__bias)
        return self.__output
        
    def adjust_delta_with(self, an_error):
        self.__delta = an_error * (self.__activation_function.dx(self.__output))
        
    def update_weight_with_input(self, inputs):
        for an_input, idx in zip(inputs, range(len(inputs))):
            self.__weights[idx] += self.__learning_rage * self.__delta * an_input
            
    def adjust_bias(self):
        self.__bias += self.__learning_rate * self.__delta
        
    def train(self, inputs, desired_output):
        self.feed(inputs)
        self.__error = desired_output - self.__output
        self.__delta = self.__error * self.__activation_function.dx(self.__output)
        for i, idx in zip(inputs, range(len(inputs))):
            self.__weights[idx] += self.__learning_rate * self.__delta * i
        self.__bias += self.__delta * self.__learning_rate
        