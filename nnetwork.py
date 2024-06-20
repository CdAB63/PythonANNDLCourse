#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 16:34:30 2024

@author: cdab63
"""

from nlayer import NLayer

class NNetwork:
    
    def __init__(self):
        self.__head = None
        self.__tail = None
        self.__layers = []
        
    def add_layer(self, layer):
        if not isinstance(layer, NLayer):
            raise ValueError(f'[ERROR] layer is {type(layer)} and not NLayer')
        if not self.__layers:
            self.__head = layer
            self.__tail = layer
            self.__layers.append(layer)
        else:
            if self.__tail.number_of_neurons() != layer.number_of_inputs():
                raise ValueError(f'[ERROR] incompatible layers {self.__tail.number_of_neurons()} vs {layer.number_of_inputs()}')
            self.__tail.set_next_layer(layer)
            layer.set_prev_layer(self.__tail)
            layer.set_next_layer(None)
            self.__tail = layer
            self.__layers.append(layer)
            
            
            