#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:27:52 2024

@author: cdab63
"""

from activation_function import SigmoidAF
from nlayer import NLayer, NLayerSpec

class NNetwork(object):
    
    def __init__(self):
        self.__layers = []
        self.__head = None
        self.__tail = None
        self.__output = None
        
    def nb_of_layers(self):
        return len(self.__layers)
    
    def layer(self, idx):
        return self.__layers[idx]
    
    def head(self):
        return self.__head
    
    def tail(self):
        return self.__tail
    
    def output(self):
        return self.__output
    
    def layer_output(self, idx):
        if not self.__head:
            raise Exception('[ERROR] empty network, no layers to return')
        layer = self.__head
        count = 0
        while count < idx:
            layer = layer.next_layer()
            count += 1
        return layer.output()
    
    def layer_n_output(self, idx):
        if not self.__head:
            raise Exception('[ERROR} empty network, no layers to return')
        return self.__layers[idx].output()

    def add_layer(self, layer):
        if not isinstance(layer, NLayer):
            raise ValueError(f'[ERROR] not a layer: {type(layer)}')
        if not self.__tail is None:
            layer.set_prev_layer(self.__tail)
            self.__tail = layer
        else:
            if not self.__head is None:
                raise ValueError('[ERROR] inconsistent network tail exists but head is None')
            layer.set_prev_layer(None)
            self.__head = layer
            self.__tail = layer
        layer.basic_set_next_layer(None)
        self.__layers.append(layer)
        
    def create_and_add_basic_layer(self, nb_of_neurons, activation_function=SigmoidAF(),
                           learning_rate=0.1, bias=0.0):
        layer = NLayer(nb_of_neurons, activation_function=activation_function,
                       learning_rate=learning_rate, bias=bias,
                       prev_layer=self.__tail)
        self.add_layer(layer)
   
    def create_network_from_json(self, file_name):
        spec = NLayerSpec()
        with open(file_name, 'r') as json_file:
            contents = json_file.readlines()
            for line in contents:
                spec.get_specs_from_string(line)
                layer = NLayer(spec.nb_of_neurons(),
                               weights_list=spec.weights_list(),
                               activation_function=spec.activation_function(),
                               learning_rate=spec.learning_rate(),
                               biases=spec.bias(), prev_layer=self.__tail)
                self.add_layer(layer)
                
    def feed(self, input_values):
        if not self.__head:
            raise Exception('[ERROR] cannot feed uninitialized neural network')
        self.__output = self.__head.feed(input_values)
        return self.__output
                
            
        
    