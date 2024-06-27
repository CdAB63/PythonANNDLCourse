#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 16:36:56 2024

@author: cdab63
"""

from activation_function import ActivationFunction, SigmoidAF
from neuron import Neuron


class NLayer(object):
    
    def __init__(self, nb_of_neurons, activation_function=SigmoidAF(),
                 bias=0.0, learning_rate=0.1, prev_layer=None, next_layer=None):
        # check arguments
        if not isinstance(nb_of_neurons, int) or nb_of_neurons < 1:
            raise ValueError('[ERROR] invalid number of neurons')
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError('[ERROR] invalid activation function')
        if not isinstance(bias, float):
            raise ValueError('[ERROR] invalid bias (should be a float)')
        if not isinstance(learning_rate, float) or learning_rate <= 0.0 or \
           learning_rate >= 1.0:
            raise ValueError('[ERROR] invalid learning rate')
        if prev_layer:
            if not isinstance(prev_layer, type(self)):
                raise ValueError('[ERROR] invalid prev_layer')
        if next_layer:
            if not isinstance(next_layer, type(self)):
                raise ValueError('[ERROR] invalid next_layer')
        # create neurons
        self.__neurons = []
        nb_of_inputs = prev_layer.number_of_neurons() if prev_layer else nb_of_neurons
        for _ in range(nb_of_neurons):
            neuron = Neuron(nb_of_inputs, activation_function=activation_function, 
                            learning_rate=learning_rate, bias=bias)
            self.__neurons.append(neuron)
        self.__nb_of_neurons = nb_of_neurons
        self.__nb_of_weights = nb_of_inputs
        self.__prev_layer = prev_layer
        self.__next_layer = next_layer
        if prev_layer:
            prev_layer.set_next_layer(self)
        if next_layer:
            next_layer.set_prev_layer(self)
        self.__activation_function = activation_function
        self.__learning_rate = learning_rate
        self.__bias = bias
        self.__error = []
        self.__output = None
        
    def prev_layer(self):
        return self.__prev_layer
    
    def set_prev_layer(self, layer):
        if layer:
            if not isinstance(layer, type(self)):
                raise ValueError('[ERROR] invalid prev_layer')
            if layer.number_of_neurons() != self.number_of_weights():
                raise ValueError('[ERROR] incompatible prev_layer')
            layer.__blind_set_next_layer(self)
        self.__prev_layer = layer

    def __blind_set_prev_layer(self, layer):
        self.__prev_layer = layer
        
    def next_layer(self):
        return self.__next_layer
    
    def set_next_layer(self, layer):
        if layer:
            if not isinstance(layer, type(self)):
                raise ValueError('[ERROR] invalid next_layer')
            if layer.number_of_weights() != self.number_of_neurons():
                raise ValueError('[ERROR] incompatible next_layer')
            layer.__blind_set_prev_layer(self)
        self.__next_layer = layer
    
    def __blind_set_next_layer(self, layer):
        self.__next_layer = layer
        
    def neurons(self):
        return self.__neurons
    
    def neuron(self, idx):
        return self.__neurons[idx]
    
    def number_of_neurons(self):
        return self.__nb_of_neurons
    
    def number_of_weights(self):
        return self.__nb_of_weights
    
    def activation_function(self):
        return self.__activation_function
    
    def set_activation_function(self, activation_function):
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError(f'[ERROR] invalid activation finction {type(activation_function)}')
        self.__activation_function = activation_function
        for neuron in self.__neurons:
            neuron.set_activation_function(activation_function)
            
    def learning_rate(self):
        return self.__learning_rate
    
    def set_learning_rate(self, value):
        if not isinstance(value, float) or value <= 0.0 or value >= 1.0:
            raise ValueError(f'[ERROR] invalid learning rate: {value}')
        self.__learning_rate = value
        for neuron in self.__neurons:
            neuron.set_learning_rate(value)
            
    def bias(self):
        return self.__bias
    
    def set_bias(self, value):
        if not isinstance(value, float):
            raise ValueError(f'[ERROR] invalid bias: {value}')
        self.__bias = value
        for neuron in self.__neurons:
            neuron.set_bias(value)
    
    def output(self):
        return self.__output
    
    def feed(self, inputs):
        self.__output = [ neuron.feed(inputs) for neuron in self.__neurons ]
        output = self.__output
        if self.__next_layer:
            output = self.__next_layer.feed(output)
        return output
        
    def feed_layer(self, inputs):
        self.__output = [ neuron.feed(inputs) for neuron in self.__neurons ]
        return self.__output