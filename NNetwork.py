#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:17:21 2024

@author: cdab63
"""

from activation_function import ActivationFunction, SigmoidAF
from nlayer import NLayer

class NNetwork:
    
    def __init__(self, layers, activation_functions=None, biases=None, learning_rates=None):
        try:
            if not all(isinstance(l, int) and l > 0 for l in layers):
                raise ValueError('[ERROR] layers is a list of neurons per layer')
        except:
            raise ValueError('[ERROR] layer must be a list')
        if activation_functions:
            try:
                if not all(issubclass(af, ActivationFunction) for af in activation_functions):
                    raise ValueError('[ERROR] invalid activation function')
            except:
                raise ValueError('[ERROR] activation_functions must be a list')
            if len(activation_functions) != len(layers):
                raise ValueError('[ERROR] layers and activation_function lists must have the same size')
        if biases:
            try:
                if not all(isinstance(b, (int, float)) for b in biases):
                    raise ValueError('[ERROR] biases must be floats')
            except:
                raise ValueError('[ERROR] biases must be a list')
            if len(biases) != len(layers):
                raise ValueError('[ERROR] layers and biases must be lists of the same size')
        if learning_rates:
            try:
                if not all(isinstance(lr, float) for lr in learning_rates):
                    raise ValueError('[ERROR] learning_rates must be a list of floats')
            except:
                raise ValueError('[ERROR] learning_rates must be a list')
            if len(learning_rates) != len(layers):
                raise ValueError('[ERROR] layers and learning_rates must be lists of the same size')
        self.__head = None
        self.__tail = None
        self.__layers = []
        for layer, idx in zip(layers, range(layers)):
            af = activation_functions[idx] if activation_functions else SigmoidAF()
            bias = biases[idx] if biases else 0.0
            lr = learning_rates[idx] if learning_rates else 0.1
            if self.__tail:
                inputs = self.__tail.number_of_neurons()
            else:
                inputs = layer
            the_layer = NLayer(layer, inputs, activation_function=af, bias=bias, learning_rate=lr)
            if self.__tail:
                the_layer.set_prev_layer(self.__tail)
            if not self.__head:
                self.__head = the_layer
            self.__tail = the_layer
            self.__layers.append(the_layer)
            
    def layers(self):
        return self.__layers
    
    def layer(self, idx):
        try:
            return self.__layers[idx]
        except:
            raise Exception('[ERROR] empty Neural Network. Define layers prior to calling this method')
            
            