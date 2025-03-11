from bandit import ExponentialSimulationDistribution, RandomPolicy
from LDPMAB import TreePolicy
import numpy as np
import time

if_test = False


def test_process_no_source():
    if if_test:

        M = 2
        n_source_list = [0] * M
        n_target = 200000
        K = 3
        dim = 2
        gamma_list = [0] * M
        epsilon_target = 2
        epsilon_list = [8] * M
        batch_size = 1

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
                            True
                            ).fit(X_source_list, arms_source_list, y_source_list)
        time_end = time.time()
        print('construction time cost', time_end - time_start, 's')


        test_point_regret, test_point_policies, _ = tree.test_single_point(mab, np.array([1 / 4, 1 / 4]), 1)
        print(test_point_regret.mean())
        unique, counts = np.unique(test_point_policies, return_counts=True)
        print(dict(zip(unique, counts)))


# def test_process_small_source():
#     if if_test:

#         M = 2
#         n_source_list = [100] * M
#         n_target = 5000
#         K = 3
#         dim = 2
#         gamma_list = [0] * M
#         epsilon_target = 8
#         epsilon_list = [8] * M
#         batch_size = 1

#         mab = ExponentialSimulationDistribution(K, dim, gamma_list)

#         X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

#         time_start = time.time()
#         tree = TreePolicy(epsilon_target,
#                             epsilon_list,
#                             n_target,
#                             n_source_list,
#                             K,
#                             dim,
#                             batch_size,
#                             ).fit(X_source_list, arms_source_list, y_source_list)
#         time_end = time.time()
#         print('construction time cost', time_end - time_start, 's')


#         test_point_regret, test_point_policies, _ = tree.test_single_point(mab, np.array([1 / 3, 1 / 3]), 1)
#         # print([test_point_regret[:(i + 1)].mean() for i in range(0, len(test_point_regret))])
#         # print([np.where(test_point_policies[:(i + 1)] == 1)[0].shape[0] / (i + 1) for i in range(0, len(test_point_policies))])
#         print(test_point_regret.mean())
#         unique, counts = np.unique(test_point_policies, return_counts=True)
#         print(dict(zip(unique, counts)))
       
# def test_process_large_source():
#     if if_test:

#         M = 2
#         n_source_list = [5000] * M
#         n_target = 20000
#         K = 3
#         dim = 2
#         gamma_list = [0] * M
#         epsilon_target = 2
#         epsilon_list = [4] * M
#         batch_size = 1
#         if_weighted = True

#         mab = ExponentialSimulationDistribution(K, dim, gamma_list)

#         X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

#         time_start = time.time()
#         tree = TreePolicy(epsilon_target,
#                             epsilon_list,
#                             n_target,
#                             n_source_list,
#                             K,
#                             dim,
#                             batch_size,
#                             if_weighted = if_weighted,
#                             ).fit(X_source_list, arms_source_list, y_source_list)
#         time_end = time.time()
#         print('construction time cost', time_end - time_start, 's')


#         test_point_regret, test_point_policies, _ = tree.test_single_point(mab, np.array([1 / 4, 1 / 4]), 1)
#         # print([test_point_regret[:(i + 1)].mean() for i in range(0, len(test_point_regret))])
#         # print([np.where(test_point_policies[:(i + 1)] == 1)[0].shape[0] / (i + 1) for i in range(0, len(test_point_policies))])
#         print(test_point_regret.mean())
#         unique, counts = np.unique(test_point_policies, return_counts=True)
#         print(dict(zip(unique, counts)))