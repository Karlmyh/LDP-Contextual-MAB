from .marginal_distributions import (
                          GammaDecayDistribution,
                          UniformDistribution,
                          )

from .reward_function import RewardFunction


from .joint_distribution import JointDistribution

import numpy as np
import math


class RandomPolicy(object):
    def __init__(self, K):
        self.K = K
    def __call__(self, X):
        return np.random.randint(self.K, size = X.shape[0]) 
    

class ImbalancedRandomPolicy(object):
    def __init__(self, K, kappa):
        self.K = K
        self.kappa = kappa
    def __call__(self, X):
        # equal diff series of length K, with minimum value be kappa / K
        probability = np.linspace(self.kappa / self.K, (2 - self.kappa) / self.K, self.K)
        probability = np.array(probability) / np.sum(probability)
        return np.random.choice(self.K, size = X.shape[0], p = probability )

class ExponentialDecayRewardFunction(object):
    def __init__(self, K, dim):
        self.K = K
        self.dim = dim
    def __call__(self, x, k):
        assert k in range(self.K)
        return 2 * np.exp( - 2 *  self.K**2 * ( x[0] - (k + 1) / (self.K + 1))**2   ) / (1 + np.exp( - 2 * self.K**2 * ( x[0] - (k + 1) / (self.K + 1))**2   ))
    
    def best_arm(self, x):
        k = np.round(x[0] * (self.K + 1)).astype(int)
        if k == 0:
            return 0
        elif k == self.K + 1:
            return self.K - 1
        else:
            return k - 1
        

class SymmetricDecayRewardFunction(object):
    def __init__(self, K, dim):
        self.K = K
        self.dim = dim
        assert dim == 2
        assert K == 2
    def __call__(self, x, k):
        assert k in range(self.K)
        return 0.5 + 0.5 * (2 * k - 1) * np.sign((x[0] - 0.5) * (x[1] - 0.5))  * (1 - 4 * np.abs([((2 * x[0]) % 1) / 2 - 0.25, ((2 * x[1]) % 1) / 2 - 0.25 ]).max() )
    
    def best_arm(self, x):
        if np.sign((x[0] - 0.5) * (x[1] - 0.5)) < 0:
            return 0
        else:
            return 1
        



class ExponentialSimulationDistribution(JointDistribution):
    def __init__(self, K, dim, gamma_list):
        self.K = K
        self.dim = dim
        self.gamma_list = gamma_list
        target_marginal_obj = UniformDistribution(np.zeros(dim), np.ones(dim))
        source_marginal_obj_list = [GammaDecayDistribution(gamma, self.dim) for gamma in self.gamma_list]
        source_policy_list = [RandomPolicy(self.K) for _ in range(len(self.gamma_list))]
        reward_function_obj = RewardFunction(ExponentialDecayRewardFunction(self.K, self.dim), self.K, self.dim)
        super(ExponentialSimulationDistribution, self).__init__(target_marginal_obj, source_marginal_obj_list, source_policy_list,  reward_function_obj)
        
class ShiftedExponentialSimulationDistribution(JointDistribution):
    def __init__(self, K, dim, gamma_list, kappa):
        self.K = K
        self.dim = dim
        self.gamma_list = gamma_list
        self.kappa = kappa
        target_marginal_obj = UniformDistribution(np.zeros(dim), np.ones(dim))
        source_marginal_obj_list = [GammaDecayDistribution(gamma, self.dim) for gamma in self.gamma_list]
        source_policy_list = [ImbalancedRandomPolicy(self.K, self.kappa) for _ in range(len(self.gamma_list))]
        reward_function_obj = RewardFunction(ExponentialDecayRewardFunction(self.K, self.dim), self.K, self.dim)
        super(ShiftedExponentialSimulationDistribution, self).__init__(target_marginal_obj, source_marginal_obj_list, source_policy_list,  reward_function_obj)
      

 

class SymmetricSimulationDistribution(JointDistribution):
    def __init__(self, K, dim, gamma_list):
        self.K = K
        self.dim = dim
        self.gamma_list = gamma_list
        target_marginal_obj = UniformDistribution(np.zeros(dim), np.ones(dim))
        source_marginal_obj_list = [GammaDecayDistribution(gamma, self.dim) for gamma in self.gamma_list]
        source_policy_list = [RandomPolicy(self.K) for _ in range(len(self.gamma_list))]
        reward_function_obj = RewardFunction(SymmetricDecayRewardFunction(self.K, self.dim), self.K, self.dim)
        super(SymmetricSimulationDistribution, self).__init__(target_marginal_obj, source_marginal_obj_list, source_policy_list,  reward_function_obj)
        
       



class ShiftedSymmetricSimulationDistribution(JointDistribution):
    def __init__(self, K, dim, gamma_list, kappa):
        self.K = K
        self.dim = dim
        self.gamma_list = gamma_list
        self.kappa = kappa
        target_marginal_obj = UniformDistribution(np.zeros(dim), np.ones(dim))
        source_marginal_obj_list = [GammaDecayDistribution(gamma, self.dim) for gamma in self.gamma_list]
        source_policy_list = [ImbalancedRandomPolicy(self.K, self.kappa) for _ in range(len(self.gamma_list))]
        reward_function_obj = RewardFunction(SymmetricDecayRewardFunction(self.K, self.dim), self.K, self.dim)
        super(ShiftedSymmetricSimulationDistribution, self).__init__(target_marginal_obj, source_marginal_obj_list, source_policy_list,  reward_function_obj)
      
