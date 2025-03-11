import numpy as np
from scipy.stats import laplace
import math


class TreeNodeEstimator(object):
    def __init__(self, 
                 epsilon_target,
                 epsilon_source_list,
                 X_range,
                 M,
                 K,
                 n_max,
                 dim,
                 max_depth,
                 node_depth,
                 batch_size,
                 warm_start_len,
                 ):
        
        self.X_range = X_range
        self.M = M
        self.K = K
        self.n_max = n_max
        self.max_depth = max_depth  
        self.epsilon_list = np.array([epsilon_target] + epsilon_source_list)
        self.dim = dim 
        self.node_depth = node_depth
        self.batch_size = batch_size
        self.tau =    2**( - node_depth / dim)
        self.warm_start_len = warm_start_len


        self.n_list = np.zeros(self.M + 1)
        self.active_flag = True
        self.active_arms = np.arange(self.K)
        self.sum_U = np.zeros((self.M + 1, self.K))
        self.sum_V = np.zeros((self.M + 1, self.K))
        self.sum_V_estimation = np.zeros((self.M + 1, self.K))
        self.sum_U_estimation = np.zeros((self.M + 1, self.K))
        self.sum_V_bound = np.zeros((self.M + 1, self.K))
        self.sum_U_bound = np.zeros((self.M + 1, self.K))
        self.lamda_numerator = np.zeros((self.M + 1, self.K))
        self.estimation = np.zeros(self.K)
        self.r = np.repeat(np.inf, self.K)
        self.lower_bound = np.zeros(self.K)
        self.upper_bound = np.zeros(self.K)


    def update(self,
               X,
               arm,
               y,
               source_id
               ):
        self.save_last_U = None
        
        if not self.active_flag:
            return False , []
        else:
            
            self.n_list[source_id] += self.batch_size
            removed_arms = []
            for k in self.active_arms:
                self.sum_U[source_id, k] += (arm == k).sum() +  laplace.rvs(size = arm.shape[0], scale = 4  / self.epsilon_list[source_id] ).sum()
                self.sum_V[source_id, k] += ((arm == k) * y).sum()  +  laplace.rvs(size = arm.shape[0], scale = 4  /  self.epsilon_list[source_id]).sum()
                self.lamda_numerator[source_id, k] = min(1,  self.epsilon_list[source_id]**2 * self.sum_U[source_id, k] / self.n_list[source_id] ) if self.n_list[source_id] > self.warm_start_len else 0
                self.sum_V_estimation[source_id, k] += self.sum_V[source_id, k] * self.lamda_numerator[source_id, k]
                self.sum_U_estimation[source_id, k] += self.sum_U[source_id, k] * self.lamda_numerator[source_id, k]
                self.sum_V_bound[source_id, k] += max(self.sum_U[source_id, k],  self.n_list[source_id] / self.epsilon_list[source_id]**2) * self.lamda_numerator[source_id, k]**2
                self.sum_U_bound[source_id, k] += self.sum_U[source_id, k] * self.lamda_numerator[source_id, k]
            
            if self.n_list[source_id] >= self.warm_start_len:
            
                self.estimation = self.sum_V_estimation.sum(axis = 0) / (self.sum_U_estimation.sum(axis = 0)+ 1e-4)
                self.estimation = np.clip(self.estimation, 0, 1)
                self.r = np.sqrt(self.sum_V_bound.sum(axis = 0) + 1e-4) / np.abs(self.sum_U_bound.sum(axis = 0))
                self.r = self.r * np.sqrt(2 * np.log2(self.n_max))
                band_radius = np.array([max(self.r[k], self.tau) for k in range(self.K)])
                self.lower_bound = self.estimation - band_radius
                self.upper_bound = self.estimation + band_radius

                if self.max_depth > self.node_depth:
                    self.active_flag = False
                    for k in self.active_arms:
                        if self.lower_bound[self.active_arms].max() > self.upper_bound[k]:
                            removed_arms.append(k)
                        if self.r[k] >= self.tau:
                            self.active_flag = True
                else:
                    for k in self.active_arms:
                        if self.lower_bound[self.active_arms].max() > self.upper_bound[k]:
                            removed_arms.append(k)
                
                self.active_arms = np.setdiff1d(self.active_arms, removed_arms)    
                if self.active_arms.shape[0] == 0:
                    print(self.lower_bound, self.upper_bound)
                    raise ValueError("No active arms, lower bound: {}, upper bound: {}, active arms: {}, removed_arms :{}".format(self.lower_bound, self.upper_bound, self.active_arms, removed_arms))
                            
            return True, removed_arms



    def visualize_estimator(self, k):
        
        for m in range(self.M):
            print("{} * {} + ".format(self.sum_V[m + 1, k],  self.lamda_numerator[m + 1, k]))
        print("{} * {}".format(self.sum_V[0, k ],  self.lamda_numerator[0, k]))
        print("-----------------------------")
        for m in range(self.M):
            print("{} * {} + ".format(self.sum_U[m + 1, k],  self.lamda_numerator[m + 1, k]))
        print("{} * {}".format(self.sum_U[0, k],  self.lamda_numerator[0, k]))




        
        
    
    
    