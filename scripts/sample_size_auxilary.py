import numpy as np
import os
import time
import scipy
import math
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

from LDPMAB import TreePolicy

from bandit import ExponentialSimulationDistribution, SymmetricSimulationDistribution



log_file_dir = "./logs/sample_size_source/"

def base_evaluate_tree_exponential(iterate, epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted):
    
    np.random.seed(iterate)
    method = "LDPMAB"
    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = n
    n_source_list = [n_m] * M
    batch_size = 500
    dim = d
    gamma_list = [gamma] * M
    

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
                        if_weighted,
                        ).fit(X_source_list, arms_source_list, y_source_list)

    test_point_regret, test_point_policies, regret_vec = tree.test_single_point(mab, np.array([1 / 3, 1 / 3]), 0)
    time_end = time.time()
    time_used = time_end - time_start

    _, counts = np.unique(test_point_policies, return_counts=True)
    test_point_arm_ratio = counts[0] / len(test_point_policies)
    regret = regret_vec.sum()
    test_regret = test_point_regret.sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                                                            "exponential",
                                                            iterate, 
                                                            epsilon, 
                                                            epsilon_m, 
                                                            gamma, 
                                                            n, 
                                                            n_m, 
                                                            d, 
                                                            M, 
                                                            K,
                                                            if_weighted,
                                                            time_used,
                                                            test_point_arm_ratio,
                                                            regret,
                                                            test_regret,
                                                            )

        f.writelines(logs)

def base_evaluate_tree_symmetric(iterate, epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted):
    
    np.random.seed(iterate)
    method = "LDPMAB"
    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = n
    n_source_list = [n_m] * M
    batch_size = 500
    dim = d
    gamma_list = [gamma] * M
    

    mab = SymmetricSimulationDistribution(K, dim, gamma_list)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source(n_source_list)

    time_start = time.time()
    tree = TreePolicy(epsilon_target,
                        epsilon_list,
                        n_target,
                        n_source_list,
                        K,
                        dim,
                        batch_size,
                        if_weighted,
                        ).fit(X_source_list, arms_source_list, y_source_list)

    test_point_regret, test_point_policies, regret_vec = tree.test_single_point(mab, np.array([1 / 3, 1 / 3]), 0)
    time_end = time.time()
    time_used = time_end - time_start

    _, counts = np.unique(test_point_policies, return_counts=True)
    test_point_arm_ratio = counts[0] / len(test_point_policies)
    regret = regret_vec.sum()
    test_regret = test_point_regret.sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".format(
                                                            "symmetric",
                                                            iterate, 
                                                            epsilon, 
                                                            epsilon_m, 
                                                            gamma, 
                                                            n, 
                                                            n_m, 
                                                            d, 
                                                            M, 
                                                            K,
                                                            if_weighted,
                                                            time_used,
                                                            test_point_arm_ratio,
                                                            regret,
                                                            test_regret,
                                                            )

        f.writelines(logs)
            
            

def run_base_auxilary():
    num_repetitions = 30
    num_jobs = 30
    
    
    # for epsilon in [1, 2]:
    #     for epsilon_m in [  1024, 2, 4, 8 , ]:
    #         for gamma in [ 0, 0.2, 1, 2]:
    #             for n in [  80000]:
    #                 for n_m in [ 0, 10000, 20000, 50000, 100000]:
    #                     for d in [2]: 
    #                         for M in [2]:
    #                             for K in [3]:
    #                                 for if_weighted in [True]:
    #                                     print("epsilon: {}, epsilon_m: {}, gamma: {}, n: {}, n_m: {}, d: {}, M: {}, K: {}, if_weighted: {}".format(epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted))
    #                                     Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_tree_exponential)(iterate, epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted) for iterate in range(num_repetitions))

    for epsilon in [1, 2]:
        for epsilon_m in [  1024, 2, 4, 8 , ]:
            for gamma in [ 0, 0.2, 1, 2]:
                for n in [  80000]:
                    for n_m in [ 0, 10000, 20000, 50000, 100000]:
                        for d in [2]: 
                            for M in [2]:
                                for K in [2]:
                                    for if_weighted in [True]:
                                        print("epsilon: {}, epsilon_m: {}, gamma: {}, n: {}, n_m: {}, d: {}, M: {}, K: {}, if_weighted: {}".format(epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted))
                                        Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_tree_symmetric)(iterate, epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted) for iterate in range(num_repetitions))
                                        
                                        