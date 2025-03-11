from bandit import ClassificationAsBandit, RandomPolicy
import numpy as np
import time

if_test = False


def test_naive_policy():
    if if_test:

        n_source_list = [20, 20]
        M = len(n_source_list)
        n_target = 20
        K = 3
        dim = 2
        reward_matrix = np.random.rand(K, K)
        X_source_list = [np.random.rand(n, dim) for n in n_source_list]
        source_policy_list = [RandomPolicy(K) for _ in range(M)]
        source_label_list = [np.random.randint(K, size = n) for n in n_source_list]
        X_target = np.random.rand(n_target, dim)
        target_label = np.random.randint(K, size = n_target)
        mab = ClassificationAsBandit(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)

        X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

        print(X_source_list, arms_source_list, y_source_list)

        X_target = mab.generate_X_target()
        policy = RandomPolicy(K)
        arms_target = policy(X_target)
        reward = mab.reward(target_label, arms_target)

        print(reward)




     