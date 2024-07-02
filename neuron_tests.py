#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 11:19:20 2024

@author: cdab63
"""

import matplotlib.pyplot as plt

import unittest

from activation_function import LinearAF, SigmoidAF
from neuron import Neuron

class NeuronTestCase(unittest.TestCase):
    
    def test_creation(self):
        neuron = Neuron(3)
        self.assertEqual(neuron.number_of_weights(), 3)
        self.assertEqual(len(neuron.weights()), 3)
        self.assertTrue(isinstance(neuron.activation_function(), SigmoidAF))
        self.assertEqual(neuron.bias(), 0.0)
        
    def test_feed(self):
        neuron = Neuron(3)
        output = neuron.feed([1.0, 1.0, 1.0])
        self.assertEqual(output, neuron.output())
        af = SigmoidAF()
        prod_vect = [ w*i for w,i in zip(neuron.weights(), [1.0, 1.0, 1.0]) ]
        test_output = af.fx(sum(prod_vect)) + neuron.bias()
        self.assertEqual(output, test_output)
        self.assertRaises(ValueError, neuron.feed, [1.0, 1.0])
        self.assertRaises(ValueError, neuron.feed, [1.0, 1.0, 1.0, 1.0])
        
    def test_feed_1(self):
        neuron = Neuron([1.0, 1.0], activation_function=LinearAF())
        output = neuron.feed([1.0, -1.0])
        self.assertEqual(neuron.output(), 0.0)
        self.assertEqual(output, 0.0)
        
    def test_af_fx(self):
        neuron = Neuron([1.0,1.0])
        output = neuron.feed([1.0, 1.0])
        res = neuron.af_fx(2)
        self.assertEqual(output, res)
        
    def test_af_dx(self):
        neuron = Neuron([1.0, 1.0])
        sigma = neuron.af_fx(2)
        derivative = sigma * (1 - sigma)
        dx = neuron.af_dx(2)
        self.assertEqual(derivative, dx)
        
    def test_error(self):
        neuron = Neuron(2)
        neuron.feed([1.0, 1.0])
        af = SigmoidAF()
        prod_vect = [w*i for w,i in zip(neuron.weights(), [1.0, 1.0]) ]
        test_output = af.fx(sum(prod_vect)) + neuron.bias()
        neuron.adjust_error(1.0)
        error = neuron.error()
        test_error = 1.0 - test_output
        self.assertEqual(error, test_error)
        
    def test_train_and(self):
        neuron = Neuron(2, learning_rate=0.1)
        for _ in range(5000):
            neuron.train([0.0, 0.0], 0.0)
            neuron.train([0.0, 1.0], 0.0)
            neuron.train([1.0, 0.0], 0.0)
            neuron.train([1.0, 1.0], 1.0)
        self.assertAlmostEqual(neuron.feed([0.0, 0.0]), 0.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([0.0, 1.0]), 0.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([1.0, 0.0]), 0.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([1.0, 1.0]), 1.0, delta=0.1)
        
    def test_train_or(self):
        neuron = Neuron(2, learning_rate=0.1)
        for _ in range(5000):
            neuron.train([0.0, 0.0], 0.0)
            neuron.train([0.0, 1.0], 1.0)
            neuron.train([1.0, 0.0], 1.0)
            neuron.train([1.0, 1.0], 1.0)
        self.assertAlmostEqual(neuron.feed([0.0, 0.0]), 0.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([0.0, 1.0]), 1.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([1.0, 0.0]), 1.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([1.0, 1.0]), 1.0, delta=0.1)

    def test_train_not(self):
        neuron = Neuron(1, learning_rate=0.1)
        for i in range(5000):
            neuron.train([0.0], 1.0)
            neuron.train([1.0], 0.0)
        self.assertAlmostEqual(neuron.feed([0.0]), 1.0, delta=0.1)
        self.assertAlmostEqual(neuron.feed([1.0]), 0.0, delta=0.1)
        
if __name__ == '__main__':
    unittest.main()