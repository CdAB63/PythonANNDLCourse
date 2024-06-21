#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 18:59:38 2024

@author: cdab63
"""

import unittest

from NNetwork import NNetwork

class NNetworkTestCase(unittest.TestCase):
    
    def test_creation(self):
        
        nn = NNetwork([2, 3, 5, 1])
        self.assertEqual(nn.number_of_layers(), 4)
        self.assertEqual(nn.layer(0).number_of_neurons(), 2)
        self.assertEqual(nn.layer(1).number_of_neurons(), 3)
        self.assertEqual(nn.layer(2).number_of_neurons(), 5)
        self.assertEqual(nn.layer(3).number_of_neurons(), 1)
        self.assertEqual(nn.head(), nn.layer(0))
        self.assertEqual(nn.tail(), nn.layer(3))
        self.assertEqual(nn.layer(0).next_layer(), nn.layer(1))
        self.assertEqual(nn.layer(3).prev_layer(), nn.layer(2))
        out = nn.feed([1, -1])
        print(f'RESULT: {out}')
        
if __name__ == '__main__':
    unittest.main()