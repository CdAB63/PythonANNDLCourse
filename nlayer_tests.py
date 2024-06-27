#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 21:28:48 2024

@author: cdab63
"""

import unittest

from nlayer import NLayer
from neuron import Neuron
from activation_function import ActivationFunction, SigmoidAF

class NLayerTestCase(unittest.TestCase):
    
    def test_creation(self):
        n1 = NLayer(2)
        n2 = NLayer(3, prev_layer=n1)
        n3 = NLayer(2, prev_layer=n2)
        self.assertEqual(n1.number_of_neurons(), 2)
        self.assertEqual(n1.number_of_weights(), 2)
        self.assertEqual(n2.number_of_neurons(), 3)
        self.assertEqual(n2.number_of_weights(), 2)
        self.assertEqual(n3.number_of_neurons(), 2)
        self.assertEqual(n3.number_of_weights(), 3)
        self.assertTrue(n1.prev_layer() is None)
        self.assertEqual(n1.next_layer(), n2)
        self.assertEqual(n2.prev_layer(), n1)
        self.assertEqual(n2.next_layer(), n3)
        self.assertEqual(n3.prev_layer(), n2)
        self.assertTrue(n3.next_layer() is None)
        
    def test_feed(self):
        n1 = NLayer(2)
        n2 = NLayer(2, prev_layer=n1)
        output = n1.feed([1.0, 1.0])
        print(f'General: {output}')
        print(f'layer 1: {n1.output()}')
        print(f'layer 2: {n2.output()}')
        output1 = n1.feed_layer([1.0, 1.0])
        print(f'Feed 1: {n1.output()}')
        output2 = n2.feed_layer(output1)
        print(f'Feed 2: {n2.output()}')
        self.assertEqual(output, output2)
        
if __name__ == '__main__':
    unittest.main()