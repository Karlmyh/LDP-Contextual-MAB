
import numpy as np


class JointDistribution(object): 
    def __init__(self, target_marginal_obj, source_marginal_obj_list, source_policy_list,  reward_function_obj, X_range = None):
        self.source_marginal_obj_list = source_marginal_obj_list
        self.source_policy_list = source_policy_list
        self.target_marginal_obj = target_marginal_obj
        self.reward_function_obj = reward_function_obj
        self.X_range = X_range
        
        if self.X_range is None:
            self.X_range = np.array([np.zeros(self.target_marginal_obj.dim),np.ones(self.target_marginal_obj.dim)])
        
        
    def generate_X_single(self, n, obj):
        X = obj.generate(n)
        return X
    
    def generate_X_target(self, n_target):
        X_target = self.generate_X_single(n_target, self.target_marginal_obj)
        return X_target

    def generate_X_source(self, n_source_list):
        X_source_list = [self.generate_X_single(n_source, obj) for n_source, obj in zip(n_source_list, self.source_marginal_obj_list)]
        return X_source_list
    
    def density_X_single(self, X, obj):
        return obj.density(X)
    
    def density_X_target(self, X):
        return self.density_X_single(X, self.target_marginal_obj)

    def density_X_source(self, X, distribution_idx):
        return self.density_X_single(X, self.source_marginal_obj_list[distribution_idx])
    
    def policy_single(self, X, policy):
        return policy(X)
    
    def policy_source(self, X_source_list):
        return [self.policy_single(X, policy) for X, policy in zip(X_source_list, self.source_policy_list)]

    def reward_function(self, X, arms):
        return self.reward_function_obj.apply(X, arms)
    
    def generate_source(self, n_source_list):
        X_source_list = self.generate_X_source(n_source_list)
        arms_source_list = self.policy_source(X_source_list)
        y_source_list = [self.reward_function(X, arms)[1] for X, arms in zip(X_source_list, arms_source_list)]
        y_source_true_list = [self.reward_function(X, arms)[0] for X, arms in zip(X_source_list, arms_source_list)]
        return X_source_list, arms_source_list, y_source_list, y_source_true_list
    
    def generate_target(self, n_target):
        X_target = self.generate_X_target(n_target)
        return X_target
    
    def intial_generate(self, n_source_list, n_target):
        X_source_list, arms_source_list, y_source_list, y_source_true_list = self.generate_source(n_source_list)
        X_target = self.generate_target(n_target)
        return X_source_list, arms_source_list, y_source_list, X_target
    
    def evaluate_regret_single(self, X, arms):
        arms = np.array([arms]).ravel()
        f_target, labels = self.reward_function(X, arms)
        f_best = self.reward_function(X, self.reward_function_obj.best_arm(X))[0]
        return (f_best - f_target).sum(), labels

    
    def evaluate_regret_static_policy(self, X, policy):
        arms = policy(X)
        regret_list = [self.evaluate_regret_single(X[i].reshape(1, -1), arms[i])[0] for i in range(X.shape[0])]
        return np.sum(regret_list)