#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 12:30:10 2024

@author: cdab63
"""

from math import exp

class ActivationFunction(object):
    
    @classmethod
    def subclasses_names(cls):
        subclasses = cls.__subclasses__()
        subclasses_names = [ c.__name__ for c in subclasses ]
        return subclasses_names
    
    def fx(self, x):
        pass
    
    def dx(self, x):
        pass
    
    
class StepAF(ActivationFunction):
    
    def fx(self, x):       
        return 1.0 if x > 0 else 0
    
    def dx(self, x):
        return 0.0
    
class LinearAF(ActivationFunction):
    
    def __init__(self, alpha=1.0):
        try:
            if alpha <= 0.0:
                raise ValueError('[ERROR] alpha should be > 0.0')
        except:
            raise ValueError('[ERROR] alpha should be a float > 0.0')
        self.__alpha = alpha
        
    def alpha(self):
        return self.__alpha
    
    def fx(self, x):
        return x * self.__alpha
    
    def dx(self, x):
        return self.__alpha

class ReLU(ActivationFunction):

    def __init__(self, alpha=1.0):
        try:
            if alpha <= 0.0:
                raise ValueError('[ERROR] alpha should be > 0.0')
        except:
            raise ValueError('[ERROR] alpha should be a float > 0.0')
        self.__alpha = alpha
        
    def alpha(self):
        return self.__alpha
    
    def fx(self, x):
        return self.__alpha * x if x > 0.0 else 0
    
    def dx(self, x):
        return self.__alpha if x > 0.0 else 0
    
class LeakyReLU(ActivationFunction):
    
        def __init__(self, alpha=1.0, beta=0.001):
            try:
                if alpha <= 0.0:
                    raise ValueError('[ERROR] alpha should be > 0.0')
            except:
                raise ValueError('[ERROR] alpha should be a float > 0.0')
            self.__alpha = alpha
            try:
                if beta <= 0.0:
                    raise ValueError('[ERROR] beta should be > 0.0')
            except:
                raise ValueError('[ERROR] beta should be a float > 0.0')
            self.__beta = beta
            
        def alpha(self):
            return self.__alpha
        
        def beta(self):
            return self.__beta
            
        def fx(self, x):
            return x * self.__alpha if x > 0 else x * self.__beta
        
        def dx(self, x):
            return self.__alpha if x > 0 else self.__beta
        
class SigmoidAF(ActivationFunction):
    
    def fx(self, x):
        return 1.0 / (1 + exp(-x))
    
    def dx(self, x):
        sigma = self.fx(x)
        return sigma * (1 - sigma)
    
    