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
        M = len(n_source_list)
        n = 200000
        K = 3
        dim = 2
        epsilon = 10
        batch_size = 100
        C = 1
        lr = 0.01
        reward_matrix = np.eye(K)
        X_source_list = [np.random.rand(n, dim) for n in n_source_list]
        source_policy_list = [RandomPolicy(K) for _ in range(M)]
        source_label_list = [np.random.randint(K, size = n) for n in n_source_list]
        X_target = np.random.rand(n, dim)
        target_label = np.array(X_target[:, 0] > 0.5).astype(int)
        mab = ClassificationAsBandit(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)

        time_start = time.time()
        linear_model = LinearPolicy(epsilon,
                            n,
                            K,
                            dim,
                            batch_size,
                            C,
                            lr, 
                            )
        time_end = time.time()
        print('construction time cost', time_end - time_start, 's')


        reward = linear_model.interaction_classification(mab)
        print(reward)


