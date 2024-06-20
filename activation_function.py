#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 14:43:31 2024

@author: cdab63
"""

from abc import ABC, abstractmethod
from math import exp

class ActivationFunction(ABC):
    
    @abstractmethod
    def fx(self, x):
        pass
    
    @abstractmethod
    def dx(self, x):
        pass
    
class LinearAF(ActivationFunction):
    
    def __init__(self, alpha=0.0, bias=0.0):
        if not isinstance(alpha, (int, float)) or alpha <= 0:
           raise ValueError(f'[ERROR] alpha should be a positive float, not {alpha}')
        if not isinstance(bias, (int, float)):
            raise ValueError(f'[ERROR] bias should be a float, not {bias}' )
        self.__alpha = alpha
        self.__bias = bias
        
    def alpha(self):
        return self.__alpha
        
    def bias(self):
        return self.__bias
    
    def fx(self, x):
        return x * self.__alpha + self.__bias
    
    def dx(self, x):
        return self.__alpha
    
class ReLU(ActivationFunction):
    
    def __init__(self, alpha=0.0):
        if not issubclass(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f'[ERROR] alpha must be float > 0, not {alpha}' )
            
        self.__alpha = alpha
        
    def alpha(self):
        return self.__alpha
    
    def fx(self, x):
        return x * self.__alpha if x > 0 else 0
    
    def dx(self, x):
        return self.__alpha if x > 0 else 0
    
class LeakyReLU(ActivationFunction):
    
    def  __init__(self, alpha=1.0, beta=0.001):
        if not issubclass(alpha, (int, float)) or alpha <= 0:
            raise ValueError(f'[ERROR] alpha must be float > 0, not {alpha}' )
        if not issubclass(beta, (int, float)) or beta < 0:
            raise ValueError(f'[ERROR] beta must be float > 0, not {beta}')
        self.__alpha = alpha
        self.__beta = beta
        
    def alpha(self):
        return self.__alpha
    
    def beta(self):
        return self.__beta
    
    def fx(self, x):
        return self.__alpha * x if x > 0 else self.__beta * x
    
    def dx(self, x):
        return self.__alpha if x > 0 else self.__beta
    
class SigmoidAF(ActivationFunction):
    
    def fx(self, x):
        return 1 / (1 + exp(-x))
    
    def dx(self, x):
        fx = self.fx(x)
        return fx * (1- fx)
    
    
    