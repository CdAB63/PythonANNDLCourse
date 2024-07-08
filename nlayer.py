#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 10:20:01 2024

@author: cdab63
"""

import json

import activation_function
from activation_function import ActivationFunction, SigmoidAF
from neuron import Neuron

class NLayerSpec(object):
    '''
    layers specs:
        list of lists
        each component:
            > number of neurons
            > optional weights list
            > optional learning rate
            > optional activation function
            > optional bias
    '''            
    def __init__(self):
        self.__nb_of_neurons = None
        self.__weights_list = None
        self.__learning_rate = 0.1
        self.__activation_function_name = 'SigmoidAF'
        self.__activation_function = SigmoidAF()
        self.__bias = 0.0
        self.__error = None
        self.__output = None
    
    def nb_of_neurons(self):
        return self.__nb_of_neurons
    
    def weights_list(self):
        return self.__weights_list
    
    def learning_rate(self):
        return self.__learning_rate
    
    def activation_function(self):
        return self.__activation_function
    
    def bias(self):
        return self.__bias
    
    def get_specs_from_string(self, spec):
        try:
            contents = json.loads(spec)
        except Exception as e:
            raise ValueError(f'[ERROR] invalid specification {str(e)}\n{spec}')
        self.__nb_of_neurons = contents['neurons']
        
        try:
            self.__weights_list = contents['weights_list']
        except:
            pass
        try:
            self.__learning_rate = contents['learning_rate']
        except:
            pass
        try:
            self.__activation_function_name = contents['activation_function']
        except:
            pass
        try:
            self.__bias = contents['bias']
        except:
            pass
        # now set the class
        try:
            cls_from_name = getattr(activation_function, self.__activation_function_name)()
            self.__activation_function = cls_from_name
        except Exception as e:
            raise ValueError(f'[ERROR] invalid activation function {self.__activation_function_name}\n {str(e)}')
        # check weights list
        if not isinstance(self.__nb_of_neurons, int) or self.__nb_of_neurons < 1:
            raise ValueError(f'[ERROR] invalid number of neurons: {self.__nb_of_neurons}')
        if not self.__weights_list is None:
            if not all(isinstance(l, list) for l in self.__weights_list):
                raise ValueError('[ERROR] invalid weights list')
            for l in self.__weights_list:
                if not all(isinstance(w, (int, float)) for w in l):
                    raise ValueError('[ERROR] invalid weights list (should be floats)')
            l_size = len(self.__weights_list[0])
            if not all(len(l) == l_size for l in self.__weights_list):
                raise ValueError('[ERROR] different weights list size for different neurons')
        return [ self.__nb_of_neurons, self.__weights_list, self.__activation_function, self.__learning_rate, self.__bias ]
        
    
class NLayer(object):
    
    def __init__(self, neurons, weights_list=None, 
                 activation_function=SigmoidAF(), 
                 learning_rate=0.1, biases=None,
                 prev_layer=None, next_layer=None):
        # Check parameters
        if not isinstance(neurons, int) or neurons < 1:
            raise ValueError(f'[ERROR] invalid number of neurons {neurons}')
        if not issubclass(type(activation_function), ActivationFunction):
            raise ValueError(f'[ERROR] invalid activation function class: {type(activation_function)}')
        # Check prev_layer & next_layer
        self.check_layer(prev_layer)
        self.check_layer(next_layer)
        # Check the learning_rate (0.0 < lr < 1.0)
        if not isinstance(learning_rate, float) or learning_rate <= 0.0 or learning_rate >= 1.0:
            raise ValueError(f'[ERROR] invalid learning_rate: {learning_rate}')
        # Set neurons parameters:
        if prev_layer:
            self.__nb_of_weights = prev_layer.nb_of_neurons()
        else:
            self.__nb_of_weights = neurons
        # if weights list is supplied, check it
        if not weights_list is None:
            if not isinstance(weights_list, list):
                raise ValueError('[ERROR] weights_list should be a list')
            if len(weights_list) != neurons:
                raise ValueError('[ERROR] weight_list has invalid length')
            if not all(isinstance(l, list) for l in weights_list):
                raise ValueError('[ERROR] invalid weights inside weights_list')
            for l in weights_list:
                if not all(isinstance(w, (int, float)) for w in l):
                    raise ValueError('[ERROR] weights must be floats')
                if len(l) != self.__nb_of_weights:
                    raise ValueError('[ERROR] invalid number of weights inside weights list')
        if weights_list:
            self.__neurons = [ Neuron(weights_list[idx], activation_function=activation_function, learning_rate=learning_rate) for idx in range(neurons) ]
        else:
            self.__neurons = [ Neuron(self.__nb_of_weights, activation_function=activation_function, learning_rate=learning_rate) for _ in range(neurons) ]

        # set the bias
        if not biases is None:
            self.set_bias(biases)
        # set the previous layer
        self.set_prev_layer(prev_layer)
        # set the next layer
        self.set_next_layer(next_layer)
        # set the output and error variables
        self.__output = []
        self.__error = []
        
    def nb_of_neurons(self):
        return len(self.__neurons)

    def nb_of_weights(self):
        return self.__nb_of_weights

    def check_layer(self, layer):
        if layer:
            if not isinstance(layer, type(self)):
                raise ValueError(f'[ERROR] Invalid layer: {type(layer)}')
                
    # Return prev and next layers
    def prev_layer(self):
        return self.__prev_layer
    
    def next_layer(self):
        return self.__next_layer
        
    # Set the prev layer and fix prev_layer next_layer  
    def basic_set_prev_layer(self, layer):
        self.check_layer(layer)
        self.__prev_layer = layer
        
    def set_prev_layer(self, prev_layer):
        if prev_layer:
            if not isinstance(prev_layer, type(self)):
                raise ValueError(f'[ERROR] Invalid prev_layer: {type(prev_layer)}')
            if prev_layer.nb_of_neurons() != self.nb_of_weights():
                raise ValueError(f'[ERROR] Incompatible layers {prev_layer.nb_of_neurons()} vs {self.__nb_of_weights}')
            self.__prev_layer = prev_layer
            prev_layer.basic_set_next_layer(self)
        else:
            self.__prev_layer = None
    
    # Set the next layer and fix next_layer prev_layer
    def basic_set_next_layer(self, layer):
        self.check_layer(layer)
        self.__next_layer = layer
        
    def set_next_layer(self, next_layer):
        if next_layer:
            if not isinstance(next_layer, type(self)):
                raise ValueError(f'[ERROR] Invalid next_layer: {type(next_layer)}')
            if self.__nb_of_neurons != next_layer.nb_of_weights():
                raise ValueError(f'[ERROR] Incompatible layers {self.__nb_of_neurons} vs {next_layer.nb_of_weights()}')
            self.__next_layer = next_layer
            next_layer.basic_set_prev_layer(self)
        else:
            self.__next_layer = None
            
    # return the output(s)     
    def output(self, idx=None):
        if not idx is None:
            try:
                return self.__output[idx]
            except:
                raise ValueError(f'[ERROR] Invalid idex: {idx}')
        else:
            return self.__output
    
    # return the error(s)
    def error(self, idx=None):
        if not idx is None:
            try:
                return self.__error[idx]
            except:
                raise ValueError(f'[ERROR] Invalid index: {idx}')
        else:
            return self.__error
        
    # set the value for errors
    def set_error(self, expected_values):
        if not self.__output:
            raise Exception('[ERROR] layer must be fed prior to error calculation')
        if self.__nb_of_neurons != len(expected_values):
            raise Exception(f'[ERROR] invalid error vector (size mismatch): {len(expected_values)}')
        self.__error = [ neuron.adjust_error(ev) for neuron, ev in zip (self.__neurons, expected_values) ]
        #self.__error = [ exp - out for exp, out in zip(expected_values, self.__output) ]

    # return the neuron(s)
    def neuron(self, idx=None):
        if idx is None:
            return self.__neurons
        else:
            try:
                return self.__neurons[idx]
            except:
                raise Exception(f'[ERROR] Invalid index {idx}')
    
    # set biases
    def set_bias(self, biases):
        if isinstance(biases, list):
            if not all(isinstance(b, (int, float)) for b in biases):
                raise ValueError('[ERROR] invalid bias (must be float)')
            if len(biases) != len(self.__neurons):
                raise ValueError('[ERROR] invalid bias list size')
            for n, b in zip(self.__neurons, biases):
                n.set_bias(b)
        elif isinstance(biases, (int, float)):
            for n in self.__neurons:
                n.set_bias(biases)
        else:
            raise ValueError('[ERROR] biases must be a list of floats of a float')
    
    # set weights
    def set_weights(self, list_of_weights):
        if not isinstance(list_of_weights, list):
            raise ValueError('[ERROR] invalid list of weights')
        if len(list_of_weights) != len(self.__neurons):
            raise ValueError('[ERROR] weights list size != neurons list size')
        for wgts in list_of_weights:
            if not all(isinstance(w, (int, float)) for w in wgts):
                raise ValueError('[ERROR] weights must be floats')
        for n, wgts in zip(self.__neurons, list_of_weights):
            n.set_weights(wgts)
            
    # feed the layer
    def feed(self, input_values):
        if not isinstance(input_values, list):
            raise ValueError(f'[ERROR] Invalid input values: {type(input_values)}')
        if not all(isinstance(inp, (int, float)) for inp in input_values):
            raise ValueError('[ERROR] Invalid input values> nan')
        if len(input_values) != self.__nb_of_weights:
            raise ValueError(f'[ERROR] Invalid input size {len(input_values)}')
        self.__output = [ self.__neurons[idx].feed(input_values) for idx in range(self.nb_of_neurons()) ]
        output = self.__output
        if self.__next_layer:
            output = self.__next_layer.feed(output)
        return output
    
    # backward error propatation
    def backward_propagate_error(self, expected_values):
        if not self.__output: # check if the network was fed
            raise Exception('[ERROR] cannot back propagate over a non fed network')
        if expected_values: # we are at the output layer
            if len(expected_values) != len(self.__output):
                raise Exception('[ERROR] size mismatch between expected and outputs')
            self.__error = []
            for ev, op, idx in zip (expected_values, self.__output, range(len(expected_values))):
                the_error = ev - op
                self.__error.append(the_error)
                self.neuron(idx).adjust_delta_with(the_error)
            if self.__prev_layer:
                self.__prev_layer.backward_propagate_error(None)
        else: # we are at hidden layer
            self.__error = []
            for neuron, idx in zip(self.__neurons, range(len(self.__neurons))):
                the_error = 0.0
                for next_neuron in self.__next_layer.neuron(None):
                    the_error += next_neuron.weight(idx) * next_neuron.delta()
                self.__error.append(the_error)
                neuron.adjust_delta_with(the_error)
            if self.__prev_layer:
                self.__prev_layer.backward_propagate_error(None)
                
    # update weights
    def update_weights(self, inputs):
        if inputs: # we are at the first layer
            inp = inputs.copy()
            for neuron in self.__neurons:
                neuron.adjust_weights(inp)
                neuron.adjust_bias()
            if self.__next_layer:
                self.__next_layer.update_weights(None)
        else: # we are in hidden or output layer
            inp = self.__prev_layer.output(None)
            self.update_weights(inp)
            