#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:27:52 2024

@author: cdab63
"""

import matplotlib.pyplot as plt

from activation_function import SigmoidAF
from nlayer import NLayer, NLayerSpec

class NNetwork(object):
    
    def __init__(self):
        self.__layers = []
        self.__head = None
        self.__tail = None
        self.__output = None
        self.__precisions = None
        self.__errors = []
        
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
                
    def calculate_average_error(self):
        if not self.__tail:
            raise Exception('[ERROR] no error for non initialized network')
        if not self.__tail.output(None):
            raise Exception('[ERROR] no error for non fed network')
        error = 0.0
        for e in self.__tail.error(None):
            error += abs(e)
        error /= len(self.__tail.error(None))
        self.__error = error
        return error
    
    def average_error(self):
        return self.__error
                
    def feed(self, input_values):
        if not self.__head:
            raise Exception('[ERROR] cannot feed uninitialized neural network')
        self.__output = self.__head.feed(input_values)
        return self.__output
    
    def backward_propagate_error(self, expected_outputs):
        self.__tail.backward_propagate_error(expected_outputs)
        
    def update_weights(self, initial_inputs):
        self.__head.update_weights(initial_inputs)
        
    def plot_average_errors(self):
        plt.plot(self.__errors, label='Average Training Errors')
        plt.xlabel('Epochs')
        plt.ylabel('Average Error')
        plt.show()
        
    def train_epoch(self, some_inputs, expected_outputs, precisions=None):
        self.feed(some_inputs)
        self.backward_propagate_error(expected_outputs)
        self.update_weights(some_inputs)
        
    def train(self, training_set, epochs=150000, calculate_errors=False, plot_errors=False, precisions=None, plot_precisions=False):
        for epoch in range(epochs):
            for training_data in training_set:
                self.train_epoch(training_data[0], training_data[1])
            if calculate_errors:
                self.calculate_average_error()
                self.__errors.append(self.average_error())
        if calculate_errors and plot_errors:
            self.plot_average_errors()