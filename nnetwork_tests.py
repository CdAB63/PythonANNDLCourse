#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:59:50 2024

@author: cdab63
"""

import unittest

from activation_function import LinearAF, SigmoidAF
from nnetwork import NNetwork
from nlayer import NLayer

class NNetworkTestCase(unittest.TestCase):
    
    def test_create_layer(self):
        nnetwork = NNetwork()
        self.assertTrue(nnetwork.head() is None)
        self.assertTrue(nnetwork.tail() is None)
        self.assertEqual(len(nnetwork.layers()), 0)
        layer = nnetwork.create_layer(3)
        self.assertTrue(isinstance(layer, NLayer))
        self.assertEqual(layer.number_of_neurons(), 3)
        layer = nnetwork.create_layer([[1.0, -1.0, 0.5], [-1.0, 1.0, 0.7], [0.5, 0.3, 2.0]])
        self.assertTrue(isinstance(layer, NLayer))
        self.assertEqual(layer.number_of_neurons(), 3)
        self.assertEqual(layer.neuron(0).weights(), [1.0, -1.0, 0.5])
        self.assertEqual(layer.neuron(1).weights(), [-1.0, 1.0, 0.7])
        self.assertEqual(layer.neuron(2).weights(), [0.5, 0.3, 2.0])
       
    def test_add_layer(self):
        nnetwork = NNetwork()
        layer = nnetwork.create_layer(3)
        nnetwork.add_layer(layer)
        self.assertEqual(nnetwork.head(), layer)
        self.assertEqual(nnetwork.tail(), layer)
        another_layer = nnetwork.create_layer(5)
        nnetwork.add_layer(another_layer)
        self.assertEqual(nnetwork.head(), layer)
        self.assertEqual(nnetwork.tail(), another_layer)
        yet_another_layer = nnetwork.create_layer(2)
        nnetwork.add_layer(yet_another_layer)
        self.assertEqual(nnetwork.head(), layer)
        self.assertEqual(nnetwork.tail(), yet_another_layer)
        
    def test_create_complex_1(self):
        nnetwork = NNetwork(layers=[ 2, 3, 1 ], activation_functions=[ SigmoidAF(), SigmoidAF(), SigmoidAF()])
        self.assertEqual(len(nnetwork.layers()), 3)
        self.assertEqual(len(nnetwork.layers()[0].neurons()), 2)
        self.assertEqual(len(nnetwork.layers()[1].neurons()), 3)
        self.assertEqual(len(nnetwork.layers()[2].neurons()), 1)
        self.assertEqual(nnetwork.layers()[0], nnetwork.head())
        self.assertEqual(nnetwork.layers()[2], nnetwork.tail())
        self.assertEqual(nnetwork.layers()[0].next_layer(), nnetwork.layers()[1])
        self.assertEqual(nnetwork.layers()[2].prev_layer(), nnetwork.layers()[1])
        self.assertEqual(nnetwork.head().prev_layer(), None)
        self.assertEqual(nnetwork.tail().next_layer(), None)
        
    def test_create_complex_2(self):
        l1 = NLayer(2)
        l2 = NLayer(3, prev_layer=l1)
        l3 = NLayer(1, prev_layer=l2)
        layers = [ l1, l2, l3 ]
        nnetwork = NNetwork(layers=layers)
        self.assertEqual(len(nnetwork.layers()), 3)
        self.assertEqual(len(nnetwork.layers()[0].neurons()), 2)
        self.assertEqual(len(nnetwork.layers()[1].neurons()), 3)
        self.assertEqual(len(nnetwork.layers()[2].neurons()), 1)
        self.assertEqual(nnetwork.layers()[0], nnetwork.head())
        self.assertEqual(nnetwork.layers()[2], nnetwork.tail())
        self.assertEqual(nnetwork.layers()[0].next_layer(), nnetwork.layers()[1])
        self.assertEqual(nnetwork.layers()[2].prev_layer(), nnetwork.layers()[1])
        self.assertEqual(nnetwork.head().prev_layer(), None)
        self.assertEqual(nnetwork.tail().next_layer(), None)
        
    def test_feed(self):
        nnetwork = NNetwork()
        l1 = nnetwork.create_layer([[1.0, 1.0], [1.0, 1.0]], activation_function=LinearAF())
        nnetwork.add_layer(l1)
        l2 = nnetwork.create_layer([[1.0, 1.0]], activation_function=LinearAF())
        nnetwork.add_layer(l2)
        output = nnetwork.feed([1.0, 1.0])
        self.assertEqual(output, [4.0])
        
    def test_feed_2(self):
        nnetwork = NNetwork(layers=[2, 3, 1], activation_functions=[ SigmoidAF(), SigmoidAF(), SigmoidAF()])
        self.assertEqual(len(nnetwork.layers()[0].neurons()), 2)
        self.assertEqual(len(nnetwork.layers()[1].neurons()), 3)
        self.assertEqual(len(nnetwork.layers()[2].neurons()), 1)
        self.assertEqual(nnetwork.head(), nnetwork.layers()[0])
        self.assertEqual(nnetwork.layers()[0].next_layer(), nnetwork.layers()[1])
        self.assertEqual(nnetwork.layers()[1].next_layer(), nnetwork.layers()[2])
        output = nnetwork.feed([1, 1])
        self.assertEqual(output, nnetwork.layers()[2].output())
        
    def test_train(self):
        nnetwork = NNetwork(layers=[2, 3, 1], activation_functions=[ LinearAF(), LinearAF(), LinearAF()])
        for count in range(20000):
            nnetwork.train([0.0, 0.0], [0.0])
            nnetwork.train([0.0, 1.0], [1.0])
            nnetwork.train([1.0, 0.0], [1.0])
            nnetwork.train([1.0, 1.0], [0.0])
        output = nnetwork.feed([0,0])
        print(f'Output: {output}')
       
        
if __name__ == '__main__':
    unittest.main()
        