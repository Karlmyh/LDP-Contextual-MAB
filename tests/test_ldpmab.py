from bandit import ExponentialSimulationDistribution, RandomPolicy
from LDPMAB import TreePolicy
import numpy as np
import time

if_test = False


def test_naive_policy():
    if if_test:

        M = 2
        n_source_list = [20000] * M
        n_target = 10000
        K = 3
        dim = 2
        gamma_list = [0] * M
        epsilon_target = 4
        epsilon_list = [4] * M
        batch_size = 100

        mab = ExponentialSimulationDistribution(K, dim, gamma_list)

        X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

        time_start = time.time()
        tree = TreePolicy(epsilon_target,
                            epsilon_list,
                            n_target,
                            n_source_list,
                            K,
                            dim,
                            batch_size,
                            ).fit(X_source_list, arms_source_list, y_source_list)
        time_end = time.time()
        print('construction time cost', time_end - time_start, 's')

        time_start = time.time()
        regret = tree.interaction(mab)
        time_end = time.time()
        print('interaction time cost', time_end - time_start, 's')
        print([regret[:(i + 1)].mean() for i in range(0, len(regret), 200)])


        test_point_regret, test_point_policies, _ = tree.test_single_point(mab, np.array([0.3, 0.3]), 1)
        # print([test_point_regret[:(i + 1)].mean() for i in range(0, len(test_point_regret))])
        # print([np.where(test_point_policies[:(i + 1)] == 1)[0].shape[0] / (i + 1) for i in range(0, len(test_point_policies))])


    
    
