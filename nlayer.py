#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:00:44 2024

@author: cdab63
"""

from neuron import Neuron
from activation_function import ActivationFunction, SigmoidAF

class NLayer:
    
    def __init__(self, nb_of_neurons, nb_of_inputs, 
                 activation_function=SigmoidAF(), 
                 bias=0.0, learning_rate=0.1):
        if not isinstance(nb_of_neurons, int) or nb_of_neurons < 1:
            raise ValueError(f'[ERROR] invalid number of neurons {nb_of_neurons}')
        self.__nb_of_neurons = nb_of_neurons
        if not isinstance(nb_of_inputs, int) or nb_of_inputs < 1:
            raise ValueError(f'[ERROR] invalid number of inputs {nb_of_inputs}')
        self.__nb_of_inputs = nb_of_inputs
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError(f'[ERROR] invalid activation function: {type(activation_function)}')
        self.__activation_function = activation_function
        if not isinstance(learning_rate, float) or learning_rate <= 0.0 or learning_rate >= 1.0:
            raise ValueError(f'[ERROR] invalid learning rate: {learning_rate}')
        if not isinstance(bias, (int, float)):
            raise ValueError(f'[ERROR] invalid bias: {bias}')
        self.__bias = bias
        self.__learning_rate = learning_rate
        self.__neurons = [ Neuron(self.__nb_of_inputs,
                                  activation_function=self.__activation_function,
                                  bias=self.__bias,
                                  learning_rate=self.__learning_rate) for _ in range(self.__nb_of_neurons) ]
        self.__prev_layer = None
        self.__next_layer = None
        self.__output = None
        self.__error = None
        
    def number_of_neurons(self):
        return self.__nb_of_neurons
    
    def number_of_inputs(self):
        return self.__nb_of_inputs
    
    def neurons(self):
        return self.__neurons
    
    def neuron(self, idx):
        return self.__neurons[idx]
   
    def prev_layer(self):
        return self.__prev_layer
    
    def next_layer(self):
        return self.__next_layer
    
    def set_prev_layer(self, layer, recurse=True):
        if layer:
            if layer == self:
                raise ValueError('[ERROR] prev_layer and self are the same')
            if not isinstance(layer, type(self)):
                raise ValueError(f'[ERROR] prev_layer is not {type(self)}')
            if self.__nb_of_inputs != layer.number_of_neurons():
                raise ValueError(f'[ERROR] prev_layer wrong size {layer.number_of_neurons()}')
            self.__prev_layer = layer
            if recurse:
                layer.set_next_layer(self, recurse=False)
        else:
            self.__prev_layer = None
            
    def set_next_layer(self, layer, recurse=True):
        if layer:
            if layer == self:
                raise ValueError('[ERROR] next_layer and self are the same')
            if not isinstance(layer, type(self)):
                raise ValueError(f'[ERROR] next_layer is not {type(self)}')
            if self.__nb_of_neurons != layer.number_of_inputs():
                raise ValueError(f'[ERROR] next_layer wrong size {layer.number_of_inputs()}')
            self.__next_layer = layer
            if recurse:
                layer.set_prev_layer(self, recurse=False)
        else:
            self.__next_layer = None

    def output(self):
        return self.__output
    
    def feed_layer(self, inputs):
        self.__output = [ n.feed(inputs) for n in self.__neurons ]
        return self.__output
        
    def feed(self, inputs):
        output = self.feed_layer(inputs)
        if self.__next_layer:
            output = self.__next_layer.feed(output)
        return output
    
    def calculate_error(self, desired_values):
        self.__error = [ d - o for d, o in zip(desired_values, self.__output) ]
        return self.__error