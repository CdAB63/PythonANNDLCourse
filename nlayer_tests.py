#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 17:15:07 2024

@author: cdab63
"""

import unittest

from activation_function import SigmoidAF, LinearAF
from nlayer import NLayer

class NLayerTestCase(unittest.TestCase):
    
    def test_create_layer(self):
        nl = NLayer(3)
        self.assertTrue(nl.nb_of_neurons() == 3)
        self.assertTrue(nl.nb_of_weights() == 3)
        self.assertTrue(isinstance(nl.neuron(idx=0).activation_function(), SigmoidAF))
        self.assertTrue(nl.neuron(idx=0).learning_rate() == 0.1)
        self.assertTrue(nl.neuron(idx=0).bias() == 0.0)
        self.assertTrue(nl.prev_layer() is None)
        self.assertTrue(nl.next_layer() is None)
        
    def test_create_layer_with_weights(self):
        nl = NLayer(3, weights_list=[ [1.0, 1.0, 1.0 ], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0] ], activation_function=LinearAF(), biases=-2.0)
        self.assertEqual(nl.nb_of_neurons(), 3)
        self.assertEqual(nl.nb_of_weights(), 3)
        self.assertTrue(nl.neuron(0).weights(), [ 1.0, 1.0, 1.0 ])
        self.assertEqual(nl.neuron(0).bias(), -2.0)
        
        
    def test_create_layers(self):
        n1 = NLayer(2)
        n2 = NLayer(3, prev_layer=n1)
        n3 = NLayer(1, prev_layer=n2)
        self.assertTrue(n2.prev_layer() == n1)
        self.assertTrue(n2.next_layer() == n3)
        self.assertTrue(n1.next_layer() == n2)
        self.assertTrue(n3.prev_layer() == n2)
        self.assertTrue(n1.prev_layer() is None)
        self.assertTrue(n3.next_layer() is None)
        
    def test_feed(self):
        n1 = NLayer(2, activation_function=LinearAF())
        n2 = NLayer(3, activation_function=LinearAF(), prev_layer=n1)
        n3 = NLayer(1, activation_function=LinearAF(), prev_layer=n2)
        output = n1.feed([1.0, 1.0])
        o1 = n1.output()
        w11 = n1.neuron(idx=0).weights()
        w12 = n1.neuron(idx=1).weights()
        eo11 = w11[0] + w11[1]
        eo12 = w12[0] + w12[1]
        self.assertEqual(o1[0], eo11)
        self.assertEqual(o1[1], eo12)
        o2 = n2.output()
        w21 = n2.neuron(idx=0).weights()
        w22 = n2.neuron(idx=1).weights()
        w23 = n2.neuron(idx=2).weights()
        eo21 = (w21[0] * o1[0]) + (w21[1] * o1[1])
        eo22 = (w22[0] * o1[0]) + (w22[1] * o1[1])
        eo23 = (w23[0] * o1[0]) + (w23[1] * o1[1])
        self.assertEqual(o2[0], eo21)
        self.assertEqual(o2[1], eo22)
        self.assertEqual(o2[2], eo23)
        o3 = n3.output()
        self.assertEqual(output[0], o3[0])
        w31 = n3.neuron(idx=0).weights()
        eo3 = (w31[0] * eo21) + (w31[1] * eo22) + (w31[2] * eo23)
        self.assertEqual(o3[0], eo3)
        
        
if __name__ == '__main__':
    unittest.main()