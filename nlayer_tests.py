#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 16:09:02 2024

@author: cdab63
"""

import unittest

from nlayer import NLayer
from activation_function import SigmoidAF, LinearAF

class NLayerTests(unittest.TestCase):
    
    def test_creation(self):
        n1 = NLayer(3, 3)
        self.assertEqual(len(n1.neurons()), 3)
        self.assertEqual(len(n1.neurons()[0].weights()), 3)
        self.assertEqual(type(n1.neurons()[0].activation_function()), SigmoidAF)
        self.assertTrue(n1.prev_layer() is None)
        self.assertTrue(n1.next_layer() is None)
        
        n2 = NLayer(5, 3)
        n2.set_prev_layer(n1)
        self.assertEqual(n1.next_layer(), n2)
        self.assertEqual(n2.prev_layer(), n1)
        
    def test_feed(self):
        n1 = NLayer(2, 2, activation_function=LinearAF())
        n2 = NLayer(3, 2, activation_function=LinearAF())
        n2.set_prev_layer(n1)
        w1 = [ n.weights() for n in n1.neurons() ]
        w2 = [ n.weights() for n in n2.neurons() ]
        output = n1.feed([1, 1])
        self.assertEqual(len(output), 3)
        o1 = [ n1.neuron(0).weight(0) + n1.neuron(0).weight(1), \
               n1.neuron(1).weight(0) + n1.neuron(1).weight(1) ]
        self.assertEqual(n1.output(), o1)
        o2 = [ n2.neuron(0).weight(0) * o1[0] + n2.neuron(0).weight(1) * o1[1],
               n2.neuron(1).weight(0) * o1[0] + n2.neuron(1).weight(1) * o1[1],
               n2.neuron(2).weight(0) * o1[0] + n2.neuron(2).weight(1) * o1[1]]
        self.assertEqual(output, o2)
        print(f'N1 Weights: {w1}')
        print(f'N2 Weights: {w2}')
        print(f'Output n1: {o1}')
        print(f'Output n2: {o2}')
        
        
if __name__ == '__main__':
    unittest.main()
        