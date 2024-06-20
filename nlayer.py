#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:00:44 2024

@author: cdab63
"""

from neuron import Neuron
from activation_function import ActivationFunction, SigmoidAF

class NLayer:
    
    def __init__(self, nb_of_neurons,nb_of_inputs, activation_function=SigmoidAF(), 
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
        self.__prev_layer = None
        if next_layer:
            if not isinstance(next_layer, type(self)):
                raise ValueError(f'[ERROR] next layer is {type(next_layer)} and not {type(self)}')
            self.__next_layer = next_layer
        else:
            self.__next_layer = None
        self.__neurons = [ Neuron(nb_of_inputs, activation_function=activation_function, bias=bias, 
                                                       learning_rate=learning_rate) for _ in range(nb_of_neurons) ]
        self.__output = None
        
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
    
    def set_prev_layer(self, layer):
        if layer:
            if not isinstance(layer, type(self)):
                raise ValueError(f'[ERROR] prev layer is {type(layer)} and not {type(self)}')
            self.__prev_layer = layer
        else:
            self.__prev_layer = None
            
    def set_next_layer(self, layer):
        if layer:
            if not isinstance(layer, type(self)):
                raise ValueError(f'[ERROR] next layer is {type(layer)} and not {type(self)}')
            self.__next_layer = layer
        else:
            self.__next_layer = None
            
    def feed(self, inputs):
        self.__output = [ n.feed(inputs) for n in self.__neurons ]