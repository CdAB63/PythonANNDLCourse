#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 10:17:09 2024

@author: cdab63
"""

import unittest

from activation_function import LinearAF
from nnetwork import NNetwork

class NNetworkTestCase(unittest.TestCase):
    
    def test_creation(self):
        nnetwork = NNetwork()
        nnetwork.create_network_from_json('network01.json')
        self.assertEqual(nnetwork.nb_of_layers(), 3)
        self.assertEqual(nnetwork.layer(0), nnetwork.head())
        self.assertEqual(nnetwork.layer(2), nnetwork.tail())
        
    def test_creation_1(self):
        nnetwork = NNetwork()
        nnetwork.create_network_from_json('network02.json')
        self.assertEqual(nnetwork.layer(0).neuron(1).weight(0), 1.0)
        
    def test_feed(self):
        nnetwork = NNetwork()
        nnetwork.create_network_from_json('network03.json')
        self.assertTrue(isinstance(nnetwork.layer(0).neuron(0).activation_function(), LinearAF))
        output = nnetwork.feed([1.0, 1.0])
        self.assertEqual(output, [ 12.0 ])
        self.assertEqual(nnetwork.layer(1).output(), [ 4.0, 4.0, 4.0 ])
        self.assertEqual(nnetwork.layer(0).output(), [ 2.0, 2.0 ])
        
    def test_train_xor(self):
        nnetwork = NNetwork()
        nnetwork.create_network_from_json('network04.json')
        training_set = [[[0.0, 0.0], [0.0]], [[0.0, 1.0], [1.0]], [[1.0, 0.0], [1.0]], [[1.0, 1.0], [0.0]]]
        nnetwork.train(training_set, calculate_errors=True, plot_errors=True)
        self.assertAlmostEqual(nnetwork.feed([0.0, 0.0])[0], 0.0, delta=0.1)
        self.assertAlmostEqual(nnetwork.feed([0.0, 1.0])[0], 1.0, delta=0.1)
        self.assertAlmostEqual(nnetwork.feed([1.0, 0.0])[0], 1.0, delta=0.1)
        self.assertAlmostEqual(nnetwork.feed([1.0, 1.0])[0], 0.0, delta=0.1)
        print()
        print(f'[0.0, 0.0]: {nnetwork.feed([0.0, 0.0])[0]}')
        print(f'[0.0, 1.0]: {nnetwork.feed([0.0, 1.0])[0]}')
        print(f'[1.0, 0.0]: {nnetwork.feed([1.0, 0.0])[0]}')
        print(f'[1.0, 1.0]: {nnetwork.feed([1.0, 1.0])[0]}')
        
if __name__ == '__main__':
    unittest.main()