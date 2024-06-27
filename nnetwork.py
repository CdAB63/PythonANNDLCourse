#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 22:34:16 2024

@author: cdab63
"""

import json

from activation_function import ActivationFunction, SigmoidAF
from nlayer import NLayer

class NNetwork(object):
    
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__learning_rate = 0.1
        self.__layers = []
        
    def head(self):
        return self.__head
    
    def tail(self):
        return self.__tail
    
    def layers(self):
        return self.__layers
    
    def layer(self, idx):
        return self.__layers[idx]
    
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
            
    def feed(self, inputs):
        self.__output = self.__head.feed(inputs)
        return self.__output