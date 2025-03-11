import numpy as np
import os
import time
import scipy
import math
import pandas as pd
from itertools import product
from joblib import Parallel, delayed
import json

from bandit import RandomPolicy
from LDPMAB import TreePolicy
from comparison.LinearBandit import LinearPolicy, NNPolicy

from bandit import ClassificationAsBanditRewardMatrix, ClassificationAsBanditTrueReward

POLICIES = {"random": RandomPolicy}


log_file_dir = "./logs/realdata/"


def base_evaluate_tree_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source_list.append(np.load("./data/{}/X_source_{}.npy".format(dataset, m)))
        source_label_list.append(np.load("./data/{}/source_label_{}.npy".format(dataset, m)))
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = target_label.shape[0]
    n_source_list = [source_label.shape[0] for source_label in source_label_list]

    method = "LDPMAB"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

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
    regret_vec = tree.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)



def base_evaluate_tree_no_source_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source_list.append(np.load("./data/{}/X_source_{}.npy".format(dataset, m)))
        source_label_list.append(np.load("./data/{}/source_label_{}.npy".format(dataset, m)))
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [] 
    n_target = target_label.shape[0]
    n_source_list = []

    method = "LDPMAB-no-source"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

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
    regret_vec = tree.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)

def base_evaluate_linear_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source_list.append(np.load("./data/{}/X_source_{}.npy".format(dataset, m)))
        source_label_list.append(np.load("./data/{}/source_label_{}.npy".format(dataset, m)))
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = target_label.shape[0]
    n_source_list = [source_label.shape[0] for source_label in source_label_list]

    method = "Linear"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

    time_start = time.time()
    linear = LinearPolicy(epsilon,
                        n_target,
                        K,
                        dim,
                        batch_size,
                        1,
                        0.01, 
                        )
    regret_vec = linear.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)



def base_evaluate_nn_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source_list.append(np.load("./data/{}/X_source_{}.npy".format(dataset, m)))
        source_label_list.append(np.load("./data/{}/source_label_{}.npy".format(dataset, m)))
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = target_label.shape[0]
    n_source_list = [source_label.shape[0] for source_label in source_label_list]

    method = "NN"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

    time_start = time.time()
    linear = NNPolicy(epsilon,
                        n_target,
                        K,
                        dim,
                        batch_size,
                        1,
                        0.01, 
                        2 * dim,
                        )
    regret_vec = linear.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)



def base_evaluate_linear_alldata_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
   
    
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source = np.load("./data/{}/X_source_{}.npy".format(dataset, m))
        X_source_list.append(X_source)
        source_label = np.load("./data/{}/source_label_{}.npy".format(dataset, m))
        source_label_list.append(source_label)
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = target_label.shape[0]
    n_source_list = [source_label.shape[0] for source_label in source_label_list]

    method = "Linear-all"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

    time_start = time.time()
    linear = LinearPolicy(epsilon,
                        n_target,
                        K,
                        dim,
                        batch_size,
                        1,
                        0.01, 
                        ).initialize(X_source_list[0], arms_source_list[0], y_source_list[0], epsilon_m)
    regret_vec = linear.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)



def base_evaluate_nn_alldata_realdata(iterate, epsilon, epsilon_m, dataset):
    
    np.random.seed(iterate)
    # read meta data and data
    meta_data = json.load(open("./data/{}/meta.txt".format(dataset)))
    K = meta_data["K"]
    dim = meta_data["dim"]
    M = meta_data["M"]
    reward_form = meta_data["reward_form"]
    source_policy_list = [POLICIES[meta_data["policy_kw"]](K)] * M
    X_target = np.load("./data/{}/X_target.npy".format(dataset))
    target_label = np.load("./data/{}/target_label.npy".format(dataset))
   
    
    X_source_list = []
    source_label_list = []
    for m in range(1, M+1):
        X_source = np.load("./data/{}/X_source_{}.npy".format(dataset, m))
        X_source_list.append(X_source)
        source_label = np.load("./data/{}/source_label_{}.npy".format(dataset, m))
        source_label_list.append(source_label)
    reward_matrix = np.load("./data/{}/reward_matrix.npy".format(dataset))

    epsilon_target = epsilon
    epsilon_list = [epsilon_m] * M
    n_target = target_label.shape[0]
    n_source_list = [source_label.shape[0] for source_label in source_label_list]

    method = "NN-all"
    batch_size = 100
    if_weighted = True

    if reward_form == "matrix":
        mab = ClassificationAsBanditRewardMatrix(K, dim, reward_matrix, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    elif reward_form == "label":
        mab = ClassificationAsBanditTrueReward(K, dim, X_source_list, source_policy_list, source_label_list, X_target, target_label)
    X_source_list, arms_source_list, y_source_list, _ = mab.generate_source()

    time_start = time.time()
    linear = NNPolicy(epsilon,
                        n_target,
                        K,
                        dim,
                        batch_size,
                        1,
                        0.01, 
                        2 * dim,
                        ).initialize(X_source_list[0], arms_source_list[0], y_source_list[0], epsilon_m)
    regret_vec = linear.interaction_classification(mab)
    time_end = time.time()
    time_used = time_end - time_start

    regret = regret_vec.sum()
    regret14 = regret_vec[0:int(len(regret_vec)/4)].sum()
    regret24 = regret_vec[0:int(len(regret_vec)/2)].sum()
    regret34 = regret_vec[0:int(len(regret_vec)*3/4)].sum()

    log_file_name = "{}.csv".format(method)
    log_file_path = os.path.join(log_file_dir, log_file_name)
    with open(log_file_path, "a") as f:
        logs= "{},{},{},{},{},{},{},{},{},{}\n".format(
                                                dataset,
                                                iterate, 
                                                epsilon, 
                                                epsilon_m,
                                                if_weighted,
                                                time_used,
                                                regret,
                                                regret14,
                                                regret24,
                                                regret34,
                                                )

        f.writelines(logs)

def run_base_realdata():
    num_repetitions = 30
    num_jobs = 30
    
    all_datasets =  ["jester", "taxi", "adult", "jobs"  ]
    datasets = ["jester", "taxi", "adult", "jobs"]


    for epsilon in [1,2,4,8,1024]:
        for epsilon_m in [   1, 2, 4, 8, 1024]:
            for dataset in datasets:
                print("epsilon: {}, epsilon_m: {}, dataset: {}".format(epsilon, epsilon_m, dataset))
                # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_tree_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))
                # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_linear_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))
                # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_nn_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))
                # Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_tree_no_source_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))
                Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_linear_alldata_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))
                Parallel(n_jobs = num_jobs, verbose = 30)(delayed(base_evaluate_nn_alldata_realdata)(iterate, epsilon, epsilon_m, dataset) for iterate in range(num_repetitions))





