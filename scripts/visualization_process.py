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



log_file_dir = "./logs/visualization_process/"

def base_evaluate_tree_exponential(params):
    
    iterate, epsilon, epsilon_m, gamma, n, n_m, d, M, K, if_weighted = params
    np.random.seed(iterate)
    method = "LDPMAB"
    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = n
    n_source_list = [n_m] * M
    batch_size = 10
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

    test_point_regret, test_point_policies, regret_vec = tree.test_single_point(mab, np.array([0.33, 0.33]), 0)
    time_end = time.time()
    time_used = time_end - time_start

    # save numpy
    filename = "{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}\n".format(
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
                                                            
                                                            )
    np.save(os.path.join(log_file_dir, "test_point_policies" + filename + ".npy"), test_point_policies)
    np.save(os.path.join(log_file_dir, "test_point_regret" + filename + ".npy"), test_point_regret)
    np.save(os.path.join(log_file_dir, "regret_vec" + filename + ".npy"), regret_vec)


            

def run_base_visualization():

    num_jobs = 50
    
    params_dic = {
        "iterate": [i for i in range(1)],
        "epsilon": [1, 2],
        "epsilon_m": [0.5, 1, 2, 8],
        "gamma": [ 0, 2, 5],
        "n": [10000],
        "n_m": [0, 100, 500, 1000],
        "d": [2],
        "M": [1],
        "K": [3],
        "if_weighted": [False]
    }
    params_list = list(product(*params_dic.values()))
    
    
    Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_tree_exponential)(params) for params in params_list)
