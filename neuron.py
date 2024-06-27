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
        self.set_activation_function(activation_function)
        self.set_bias(bias)
        self.set_learning_rate(learning_rate)
        if isinstance(weights, int):
            self.set_random_weights(weights)
        else:
            self.set_listed_weights(weights)
        self.__output = None
        self.__error  = None
        self.__delta  = None
        
    def set_random_weights(self, number, min_max=(-1.0, 1.0)):
        if not isinstance(number, int) or number < 1:
            raise ValueError(f'[ERROR] invalid number of weights: {number}')
        self.__weights = [ random.uniform(min_max[0], min_max[1]) for _ in range(number)]
        
    def set_listed_weights(self, the_weights):
        if not all(isinstance(weight, float) for weight in the_weights):
            raise ValueError('[ERROR] all weights must be floats')
        self.__weights = the_weights
        
    def number_of_weights(self):
        return len(self.__weights)
    
    def weights(self):
        return self.__weights
    
    def weight(self, idx):
        return self.__weights[idx]

    def set_weight(self, idx, value):
        if not isinstance(value, float) or not isinstance(idx, int) or idx < 0:
            raise ValueError(f'[ERROR] invalid parameters {idx}, {value}')
        self.__weights[idx] = value
        
    def set_weights(self, new_weights):
        if len(new_weights) != len(self.__weights):
            raise ValueError('[ERROR] different dimensions of current weights and new_weights')
        self.__weights = new_weights.copy()
    
    def activation_function(self):
        return self.__activation_function
    
    def af_fx(self, value):
        return self.__activation_function.fx(value)
    
    def af_dx(self, value):
        return self.__activation_function.dx(value)
    
    def set_activation_function(self, activation_function):
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError(f'[ERROR] invalid activation function: {type(activation_function)}')
        self.__activation_function = activation_function
        
    def learning_rate(self):
        return self.__learning_rate
    
    def set_learning_rate(self, value):
        if not isinstance(value, float) or value <= 0.0 or value >= 1.0:
            raise ValueError(f'[ERROR] invalid learning rate: {value}')
        self.__learning_rate = value
        
    def bias(self):
        return self.__bias
    
    def set_bias(self, value):
        if not isinstance(value, float):
            raise ValueError(f'[ERROR] invalid bias: {value}')
        self.__bias = value
        
    def output(self):
        return self.__output
    
    def delta(self):
        return self.__delta
    
    def error(self):
        return self.__error
    
    def feed(self, values):
        if len(values) != len(self.__weights):
            raise ValueError(f'[ERROR] inputs and weights should have same dimension but: {len(values)} vs {len(self.__weights)}')
        self.__output = self.af_fx(sum([w * i for w, i in zip(self.weights(), values)])) + self.__bias
        return self.__output
        
    # Training stuff
    def adjust_error(self, expected_value):
        self.__error = expected_value - self.__output
        
    def adjust_delta(self):
        self.__delta = self.error() * self.af_dx(self.output())
        
    def adjust_weights(self, values):
        if len(values) != len(self.weights()):
            raise ValueError('[ERROR] incompatible inputs vs weights vector sizes')
        self.__weights = [ weight + (self.learning_rate() * self.delta() * value) \
                           for weight, value in zip(self.weights(), values) ]  
            
    def adjust_bias(self):
        self.__bias += self.learning_rate() * self.delta()
        
    def train(self, values, expected_output):
        self.feed(values) # feed the values
        self.adjust_error(expected_output) # set the error
        self.adjust_delta() # adjust weights correction parameter
        self.adjust_weights(values) # adjust the weights
        self.adjust_bias() # adjust bias
        
        
        
        
        
    
    