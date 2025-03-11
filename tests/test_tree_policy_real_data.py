from bandit import ExponentialSimulationDistribution, RandomPolicy
from comparison.LinearBandit import NNPolicy, LinearPolicy
from LDPMAB import TreePolicy
import numpy as np
import time

if_test = True

from bandit import ClassificationAsBandit, RandomPolicy
import numpy as np
import time

if_test = False





     

def test_process_no_source():
    if if_test:

        n_source_list = [20, 20]
        epsilon_list = [8, 8]
        M = len(n_source_list)
        n = 20000
        K = 3
        dim = 2
        epsilon = 10
        batch_size = 100
        reward_matrix = np.eye(K)
        X_source_list = [np.random.rand(n, dim) for n in n_source_list]
        source_policy_list = [RandomPolicy(K) for _ in range(M)]
        source_label_list = [np.random.randint(K, size = n) for n in n_source_list]
        X_target = np.random.rand(n, dim)
        target_label = np.array(X_target[:, 0] > 0.5).astype(int)
        mab = ClassificationAsBandit(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)

        X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

        time_start = time.time()
        tree = TreePolicy(epsilon,
                            epsilon_list,
                            n,
                            n_source_list,
                            K,
                            dim,
                            batch_size,
                            ).fit(X_source_list, arms_source_list, y_source_list)
        time_end = time.time()
        print('construction time cost', time_end - time_start, 's')


        reward = tree.interaction_classification(mab)
        print(reward)


