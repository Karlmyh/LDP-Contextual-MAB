#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 28 09:17:14 2021

@author: mayuheng
"""



import numpy as np

class Distribution(object): 
    def __init__(self):
        pass
    def generate(self, num_samples):
        sample_X = self.sampling(num_samples) 
    
        return sample_X
    def sampling(self, num_samples): 
        pass
    def density(self, sample_X): 
        pass

class UniformDistribution(Distribution): 
    def __init__(self, low, upper):
        super(UniformDistribution, self).__init__()
        low = np.array(low).ravel()
        upper = np.array(upper).ravel()
        self.dim = len(low)
        self.low = low
        self.upper = upper
        
    def sampling(self, num_samples):
        return np.random.rand(num_samples,self.dim) * (self.upper-self.low) + self.low
    
    def density(self, sample_X):
        in_interval_low=np.array([(sample_X[i]>=self.low).all() for i in range(len(sample_X))])
        in_interval_up=np.array([(sample_X[i]<=self.upper).all() for i in range(len(sample_X))])
        return in_interval_low * in_interval_up / np.prod(self.upper - self.low)


class GammaDecayDistribution(Distribution):
    def __init__(self, gamma, dim):
        super(GammaDecayDistribution, self).__init__()
        self.dim = dim
        self.gamma = gamma
    
    def sampling(self, num_samples):
        X = np.random.rand(num_samples, self.dim)
        u = np.random.rand(num_samples)
        scale = u**(1 / self.dim / (self.gamma + 1)) / 2 / np.abs(X - 1 / 2).max(axis = 1)
        return (X - 1 / 2) * scale[:, None] + 1 / 2
    
    def density(self, sample_X):
        return  np.abs(sample_X - 1 / 2).max(axis = 1)**self.gamma / self.dim * (self.dim + self.gamma) * 2**(self.gamma)
   



  