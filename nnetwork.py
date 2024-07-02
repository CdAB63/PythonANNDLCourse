#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:34:16 2024

@author: cdab63
"""

from activation_function import ActivationFunction, SigmoidAF
from nlayer import NLayer

class NNetwork(object):
    
    def __init__(self, layers=None, activation_functions=None, learning_rate=0.1):
        self.__head = None
        self.__tail = None
        self.__learning_rate = learning_rate
        self.__layers = []
        if layers is None:
            pass
        elif all(isinstance(item, NLayer) for item in layers):
            for layer in layers:
                if self.__tail:
                    layer.set_prev_layer(self.__tail)
                self.add_layer(layer)
        elif all(isinstance(item, int) for item in layers) and \
             all(issubclass(type(item), ActivationFunction) for item in activation_functions) and \
             len(layers) == len(activation_functions):
                 for l, af in zip(layers, activation_functions):
                     if self.__tail:
                         layer = NLayer(l, activation_function=af, learning_rate=learning_rate, prev_layer=self.__tail)
                     else:
                         layer = NLayer(l, activation_function=af, learning_rate=learning_rate)
                     self.add_layer(layer)
        else:
            raise ValueError('[ERROR] invalid parameters')
        self.__errors = []
        self.__precisions = []
        
    '''
        ACCESSOR METHODS
    '''
    
    def head(self):
        return self.__head
    
    def tail(self):
        return self.__tail
    
    def layers(self):
        return self.__layers
    
    def layer(self, idx):
        return self.__layers[idx]
    
    def errors(self):
        return self.__errors
    
    def precisions(self):
        return self.__precisions

    '''
        NEURAL NETWORK SETUP METHODS
    '''
    
    def create_layer(self, neurons, activation_function=SigmoidAF(),
                     bias=0.0, learning_rate=0.1):
        if isinstance(neurons, int):
            layer = NLayer(neurons, 
                           activation_function=activation_function, 
                           bias=bias, learning_rate=learning_rate,
                           prev_layer=self.__tail)
        else:
            layer = NLayer(len(neurons),
                           activation_function=activation_function,
                           bias=bias, learning_rate=learning_rate,
                           prev_layer=self.__tail)
            for w, n in zip(neurons, layer.neurons()):
                n.set_weights(w)
                
        return layer
    

    def add_layer(self, layer):
        if self.__layers:
            if not self.__head or not self.__tail:
                raise ValueError('[ERROR] inconsistent layers list')
            if not layer.prev_layer() == self.__tail:
                raise ValueError('[ERROR] layer.prev_layer and self.__tail do not match')
            self.__layers.append(layer)
            self.__tail = layer
        else:
            if self.__head or self.__tail:
                raise ValueError('[ERROR] inconsistent layers list')
            if layer.prev_layer():
                raise ValueError('[ERROR] layer cannot be inserted in empty layers list if has prev_layer')
            self.__layers.append(layer)
            self.__head = layer
            self.__tail = layer
    
    '''
        NEURAL NETWORD EVALUATION (FEED) METHODS
    '''
    
    def feed(self, inputs):
        self.__output = self.__head.feed(inputs)
        return self.__output
    
    '''
        TRAINING METHODS
    '''
    
    def backward_propagate_error(self, expected_outputs):
        self.__tail.backward_propagate_error(expected_outputs)
        
    def update_weights(self, initial_inputs):
        self.__head.update_weights(initial_inputs)
        
    def train(self, some_inputs, desired_outputs):
        self.feed(some_inputs)
        self.backward_propagate_error(desired_outputs)
        self.update_weights(some_inputs)