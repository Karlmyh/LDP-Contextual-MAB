# a class to utilize classification dataset to produce bandit data

import numpy as np
import math


class ClassificationAsBanditRewardMatrix(object):
    def __init__(self, K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label):
        self.K = K
        self.dim = dim
        self.reward_matrix = reward_matrix
        self.X_source_list = X_source_list  
        self.source_policy_list = source_policy_list
        self.source_label_list = source_label_list
        self.X_target = X_target
        self.target_label = target_label


    def policy_single(self, X, policy):
        return policy(X)

    def policy_source(self, X_source_list):
        return [self.policy_single(X, policy) for X, policy in zip(X_source_list, self.source_policy_list)]

    def reward(self, true_arms, arms):
        return self.reward_matrix[true_arms, arms]

    def generate_source(self, n_source_list = None):
        arms_source_list = self.policy_source(self.X_source_list)
        y_source_list = [self.reward(true_arms, arms) for true_arms, arms in zip(self.source_label_list, arms_source_list)]
        return self.X_source_list, arms_source_list, y_source_list, None
    
    def generate_X_target(self, n_target = None):
        return self.X_target

    def generate_target(self, n_target = None):
        return self.generate_X_target()
    
    def evaluate_regret_single_classification(self, idx, arms):
        true_labels = self.target_label[idx]
        return self.reward(true_labels, arms)



class ClassificationAsBanditTrueReward(object):
    def __init__(self, K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_labels):
        self.K = K
        self.dim = dim
        self.X_source_list = X_source_list  
        self.source_policy_list = source_policy_list
        self.source_label_list = source_label_list
        self.X_target = X_target
        self.target_labels = target_labels


    def policy_single(self, X, policy):
        return policy(X)

    def policy_source(self, X_source_list):
        return [self.policy_single(X, policy) for X, policy in zip(X_source_list, self.source_policy_list)]

    def reward(self, idx, arms):
        return self.target_labels[idx, arms]

    def generate_source(self, n_source_list = None):
        arms_source_list = self.policy_source(self.X_source_list)
        y_source_list = [self.reward(np.arange(souce_label.shape[0]), arms) for souce_label, arms in zip(self.source_label_list, arms_source_list)]
        return self.X_source_list, arms_source_list, y_source_list, None
    
    def generate_X_target(self, n_target = None):
        return self.X_target

    def generate_target(self, n_target = None):
        return self.generate_X_target()
    
    def evaluate_regret_single_classification(self, idx, arms):
        return self.reward(idx, arms)