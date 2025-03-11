from bandit import ExponentialSimulationDistribution, RandomPolicy
from LDPMAB import TreePolicy
import numpy as np
import time

if_test = False



def test_process_compare_small_and_large_source():
    if if_test:

        M = 2
        n_source_list = [20000] * M
        n_target = 80000
        K = 3
        dim = 2
        gamma_list = [0] * M
        epsilon_target = 4
        epsilon_list = [8] * M
        batch_size = 100
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

        test_point_regret, test_point_policies, _ = tree.test_single_point(mab, np.array([0.25, 0.25]), 1)
        nodeid = tree.tree_.apply_active(np.array([[0.25, 0.25]]))
        node = tree.tree_.leafnode_fun[nodeid.item()]
        print("depth", node.node_depth)
        print("sum U", node.sum_U)
        print("sum V", node.sum_V)
        print("lamda_numerator", node.lamda_numerator)
        print("estimation", node.estimation)
        print("r", node.r)
        print("active_arms", node.active_arms)


        





       

       