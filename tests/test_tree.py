from bandit import ExponentialSimulationDistribution, RandomPolicy
from LDPMAB import TreePolicy
import numpy as np
import time

if_test = False


def test_naive_policy():
    if if_test:

        M = 2
        n_source_list = [100000] * M
        n_target = 5000
        K = 3
        dim = 2
        gamma_list = [0] * M
        epsilon_target = 1024
        epsilon_list = [1024] * M
        batch_size = 1

        mab = ExponentialSimulationDistribution(K, dim, gamma_list)

        X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

        tree = TreePolicy(epsilon_target,
                            epsilon_list,
                            n_target,
                            n_source_list,
                            K,
                            dim,
                            batch_size,
                            ).fit(X_source_list, arms_source_list, y_source_list)
        regret = tree.interaction(mab)

        print(regret)
        
        x = np.array([1 / 3, 1 / 3]).reshape(1, -1)

        leaf_node = tree.tree_.apply(x)
        print(leaf_node)
        print(tree.tree_.get_all_anscestor(x))
        print(tree.tree_.get_node_range(leaf_node))
        print(tree.tree_.get_all_descendant(1))

        tree.visualize_path(x)

        




    
    
