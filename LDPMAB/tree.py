import numpy as np

from ._tree import TreeStruct
from ._splitter import MaxEdgeRandomSplitter

from ._estimator import TreeNodeEstimator







class TreePolicy(object):
    """ Abstact Recursive Tree Structure.    
    """
    def __init__(self, 
                 epsilon_target,
                 epsilon_list,
                 n_target,
                 n_source_list,
                 K,
                 dim,
                 batch_size,
                 if_weighted = True,
                ):
        
        self.splitter = MaxEdgeRandomSplitter
        self.estimator = TreeNodeEstimator
        self.epsilon_target = epsilon_target
        self.epsilon_list = epsilon_list
        self.n_target = n_target
        self.n_source_list = n_source_list
        self.K = K
        self.dim = dim
        self.batch_size = batch_size
        self.if_weighted = if_weighted

        
        
        self.X_range = np.array([np.zeros(self.dim),np.ones(self.dim)])
        self.M = len(n_source_list)
        self.tree_ = None
        depth_vec = [min(np.log2(2 * n * (self.M + 1)) * dim / (2 + dim), np.log2(2 * n * epsilon**2 * (self.M + 1) ) * dim / (2 + 2 * dim))  if n > 0 else -1 for epsilon, n in zip(epsilon_list + [epsilon_target] , n_source_list + [n_target] ) ]
        effective_sample_size_vec = [ n * min(1, epsilon)**2  for epsilon, n in zip(epsilon_list + [epsilon_target] , n_source_list + [n_target] ) ]
        self.max_depth = int(np.max(depth_vec))
        self.n_max = np.max(effective_sample_size_vec)
        self.warm_start_len = int(np.log2(self.n_max))**2

             
    def fit(self, X_source_list, arms_source_list, y_source_list):
        self.tree_ = TreeStruct(self.splitter, 
                                self.estimator, 
                                self.max_depth,
                                self.epsilon_target,
                                self.epsilon_list,
                                self.n_target,
                                self.n_source_list,
                                self.n_max,
                                self.K,
                                self.M,
                                self.dim,
                                self.X_range,
                                self.batch_size,
                                self.warm_start_len)
                            
        self.tree_.initialize()

        for m in range(self.M):
            batch_num = X_source_list[m].shape[0] // self.batch_size
            for i in range(batch_num):
                X = X_source_list[m][i * self.batch_size:(i + 1) * self.batch_size]
                arms = arms_source_list[m][i * self.batch_size:(i + 1) * self.batch_size]
                label = y_source_list[m][i * self.batch_size:(i + 1) * self.batch_size]
                self.update(X, arms, label, m + 1)
        return self
    

        
    def policy(self, X):
        if self.if_weighted:
            return np.array(self.tree_.weighted_policy(X))
        else:
            return np.array(self.tree_.policy(X))

    def update(self, X, arms, y, source_id):
        self.tree_.apply_update(X, arms, y, source_id)
        return self
    

    def interaction(self, mab):
        X_target = mab.generate_target(self.n_target)
        batch_num = X_target.shape[0] // self.batch_size
        regret_vec = np.zeros(batch_num)
        for i in range(batch_num):
            X = X_target[i * self.batch_size:(i + 1) * self.batch_size]
            arms = self.policy(X)
            regret, label = mab.evaluate_regret_single(X, arms)
            self.update(X, arms, label, 0)
            regret_vec[i] = regret
        return regret_vec
    
    def interaction_classification(self, mab):
        X_target = mab.generate_target(self.n_target)
        batch_num = X_target.shape[0] // self.batch_size
        reward_vec = np.zeros(batch_num)
        for i in range(batch_num):
            idx = np.arange(i * self.batch_size, (i + 1) * self.batch_size)
            X = X_target[i * self.batch_size:(i + 1) * self.batch_size]
            arms = self.policy(X)
            reward = mab.evaluate_regret_single_classification(idx, arms)
            self.update(X, arms, reward, 0)
            reward_vec[i] = reward.sum()
        return reward_vec
                 
    def test_single_point(self, mab, x, if_visualize):
        X_target = mab.generate_target(self.n_target)
        batch_num = X_target.shape[0] // self.batch_size
        regret_vec = np.zeros(batch_num)
                 
        test_regret = np.zeros(batch_num)
        test_policy = np.zeros(batch_num)
        x = x.reshape(1, -1)
        for i in range(batch_num):
            X = X_target[i * self.batch_size:(i + 1) * self.batch_size]
            arms = self.policy(X)
            test_policy[i] = self.policy(x)
            regret, label = mab.evaluate_regret_single(X, arms)
            regret_vec[i] = regret
            test_regret[i], _ = mab.evaluate_regret_single(x, test_policy[i])
            self.update(X, arms, label, 0)

            if if_visualize:
                if i % (batch_num // 40) == (batch_num // 40) - 1:
                    print('{0:*^80}'.format("visualize"))
                    self.visualize_path(x)
                    # self.visualize_estimator( x, mab.reward_function_obj.best_arm(x).ravel()[0])
        return test_regret, test_policy, regret_vec

    
                
        
    def apply(self, X):
        """Reture the belonging cell ids. 
        """
        return self.tree_.apply(X)
    
    
    def get_node_idx(self,X):
        """Reture the belonging cell ids. 
        """
        return self.apply(X)
    
    def get_node(self,X):
        """Reture the belonging node. 
        """
        return [self.tree_.leafnode_fun[i] for i in self.get_node_idx(X)]
    
    def get_all_node(self):
        """Reture all nodes. 
        """
        return list(self.tree_.leafnode_fun.values())
    
    def get_all_node_idx(self):
        """Reture all node indexes. 
        """
        return list(self.tree_.leafnode_fun.keys())

    def visualize_path(self, x):
        """Visualize the tree structure. 
        """
        return self.tree_.visualize_path(x)
    
    def visualize_estimator(self, x, k):
        """Visualize the estimator. 
        """
        return self.tree_.visualize_estimator(x, k)

    def visualize_bound(self, x, k):
        """Visualize the estimator. 
        """
        return self.tree_.visualize_estimator(x, k)
    
    
    
    

