#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 15:42:41 2024

@author: cdab63
"""

import unittest
from neuron import Neuron

class NeuronTestCase(unittest.TestCase):
    
    def test_train_and(self):
        n = Neuron(2)
        for _ in range(50000):
            n.train([0, 0], 0)
            n.train([0, 1], 0)
            n.train([1, 0], 0)
            n.train([1, 1], 1)
        self.assertAlmostEqual(n.feed([0,0]), 0.0, delta=0.01)
        self.assertAlmostEqual(n.feed([0,1]), 0.0, delta=0.01)
        self.assertAlmostEqual(n.feed([1,0]), 0.0, delta=0.01)
        self.assertAlmostEqual(n.feed([1,1]), 1.0, delta=0.01)

    def test_train_or(self):
        n = Neuron(2)
        for _ in range(50000):
            n.train([0, 0], 0)
            n.train([0, 1], 1)
            n.train([1, 0], 1)
            n.train([1, 1], 1)        
        self.assertAlmostEqual(n.feed([0,0]), 0.0, delta=0.01)
        self.assertAlmostEqual(n.feed([0,1]), 1.0, delta=0.01)
        self.assertAlmostEqual(n.feed([1,0]), 1.0, delta=0.01)
        self.assertAlmostEqual(n.feed([1,1]), 1.0, delta=0.01)      
if __name__ == '__main__':
    
    unittest.main()