#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 10:06:30 2024

@author: cdab63
"""

import random

from activation_function import ActivationFunction, SigmoidAF

class Neuron(object):
    
    def __init__(self, weights, activation_function=SigmoidAF(), bias=0.0,
                 learning_rate=0.1):
        self.set_weights(weights)
        self.set_activation_function(activation_function)
        self.set_bias(bias)
        self.set_learning_rate(learning_rate)
        self.__output = None
        self.__error = None
        self.__delta = None
        
    '''
    ACCESSOR METHODS
    '''

    def weights(self):
        return self.__weights
    
    def weight(self, idx):
        return self.__weights[idx]
    
    def set_random_weights(self, number_of_weights):
        if number_of_weights < 1:
            raise ValueError(f'[ERROR] invalid number of weights: {number_of_weights}')
        self.__weights = [ random.uniform(-1.0, 1.0) for _ in range(number_of_weights) ]
        
    def set_listed_weights(self, listed_weights):
        if not isinstance(listed_weights, list):
            raise ValueError('[ERROR] invalid weights (not a list)')
        if not all(isinstance(item, (int, float)) for item in listed_weights):
            raise ValueError('[ERROR] not all weights are floats')
        self.__weights = listed_weights.copy()
        
    def set_weights(self, weights):
        if isinstance(weights, int):
            self.set_random_weights(weights)
        else:
            self.set_listed_weights(weights)
            
    def number_of_weights(self):
        return len(self.__weights)

            
    def activation_function(self):
        return self.__activation_function
    
    def af_fx(self, x):
        return self.__activation_function.fx(x)
        
    def af_dx(self, x):
        return self.__activation_function.dx(x)
    
    def set_activation_function(self, af):
        if not issubclass(type(af), ActivationFunction):
            raise ValueError(f'[ERROR] not a valid activation function: {type(af)}')
        self.__activation_function = af
        
    def bias(self):
        return self.__bias
    
    def set_bias(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError(f'[ERROR] invalid bias: {value}')
        self.__bias = value
        
    def learning_rate(self):
        return self.__learning_rate
    
    def set_learning_rate(self, value):
        if not isinstance(value, float) or value <= 0.0 or value >= 1.0:
            raise ValueError(f'[ERROR] invalid learning rate: {value}')
        self.__learning_rate = value
        
    def output(self):
        return self.__output
    
    def error(self):
        return self.__error
    
    def delta(self):
        return self.__delta
        
    '''
    FEED
    '''
    def feed(self, inputs):
        if not isinstance(inputs, list) or len(inputs) != len(self.__weights):
            raise ValueError('[ERROR] invalid inputs: {inputs}')
        self.__output = self.af_fx(sum([ w * i for w, i in zip(self.__weights, inputs)]) + self.__bias)
        return self.__output
    
    '''
    TRAIN STUFF
    '''
    
    def adjust_error(self, value):
        self.__error = value - self.__output
        
    def adjust_delta(self):
        self.__delta = self.__error * self.af_dx(self.__output)
        
    def adjust_delta_with(self, an_error):
        self.__delta = an_error * self.af_dx(self.__output)
        
    def adjust_bias(self):
        self.__bias += self.__learning_rate * self.__delta
        
    def adjust_weights(self, inputs):
        for an_input, idx in zip(inputs, range(len(inputs))):
            self.__weights[idx] += self.__learning_rate * self.__delta * an_input
        
    def train(self, inputs, desired_output):
        self.feed(inputs)
        self.adjust_error(desired_output)
        self.adjust_delta()
        self.adjust_weights(inputs)
        self.adjust_bias()
        
        
        
                                   