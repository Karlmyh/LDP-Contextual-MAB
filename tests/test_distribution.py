from bandit import ExponentialSimulationDistribution, RandomPolicy, SymmetricSimulationDistribution
import numpy as np
import time

if_test = False

def test_generation():
    if if_test:

        n_source_list = [100, 100]
        n_target = 100
        K = 3
        dim = 2
        gamma_list = [0,1]
        mab = ExponentialSimulationDistribution(K, dim, gamma_list)

        X_source_list, arms_source_list, y_source_list, X_target = mab.intial_generate(n_source_list, n_target)

        # print(X_source_list, arms_source_list, y_source_list, X_target)

def test_generation_time():
    if if_test:

        n_source_list = [100, 100]
        n_target = 1000000
        K = 3
        dim = 2
        gamma_list = [0,1]
        mab = ExponentialSimulationDistribution(K, dim, gamma_list)

        time_start = time.time()
        X_target = mab.generate_target(n_target)
        time_end = time.time()
        print('time cost together', time_end-time_start, 's')

        time_start = time.time()
        for i in range(n_target // 20):
            X_target = mab.generate_target(20)
        time_end = time.time()
        print('time cost separate', time_end-time_start, 's')
        
        # print(X_source_list, arms_source_list, y_source_list, X_target)


def test_naive_policy():
    if if_test:

        n_source_list = [100, 100]
        n_target = 100
        K = 3
        dim = 2
        gamma_list = [0,1]
        mab = ExponentialSimulationDistribution(K, dim, gamma_list)

        X_source_list, arms_source_list, y_source_list, X_target = mab.intial_generate(n_source_list, n_target)

        time_start = time.time()
        regret = 0
        for x in X_target:
            regret += mab.evaluate_regret_single(x.reshape(1, -1), np.random.randint( K))[0]
        print(regret)
        time_end = time.time()
        print('time cost iterative', time_end-time_start, 's')

        time_start = time.time()
        policy = RandomPolicy(K)
        mab.evaluate_regret_static_policy(X_target, policy)
        time_end = time.time()
        print('time cost static', time_end-time_start, 's')


        time_start = time.time()
        # batch evaluation
        print(mab.evaluate_regret_single(X_target, np.random.randint( K, size = n_target))[0])
        time_end = time.time()
        print('time cost batch', time_end-time_start, 's')
    


def test_symmetric_distribution():

    if if_test:
        n_source_list = [100, 100]
        n_target = 100
        K = 2
        dim = 2
        gamma_list = [0,1]
        mab = SymmetricSimulationDistribution(K, dim, gamma_list)

        X_source_list, arms_source_list, y_source_list, X_target = mab.intial_generate(n_source_list, n_target)

        for y_source in y_source_list:
            print(y_source.max(), y_source.min())
     