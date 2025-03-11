import numpy as np


class RewardFunction(object): 
    def __init__(self, func, K, dim):
        self.func = func
        self.dim = dim
        self.K = K
        try:
            for k in range(self.K ):
                x = np.random.rand(self.dim)
                y = self.func(x, k)
        except:
            raise ValueError("f should receive {} dimensional numpy ndarray".format(self.dim))
        assert type(y) in [float, int, np.float64, np.float32, 
                           np.float16, np.int64, np.int32, np.int16, np.int8]

    def apply(self, X, arms):
        if type(arms) in [int, np.int64, np.int32, np.int16, np.int8]:
            arms = np.array([arms])
        assert X.shape[0] == arms.shape[0]
        reward_function_value = np.zeros(X.shape[0])
        reward = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            reward_function_value[i] = self.func(X[i], arms[i])
            reward[i] = np.random.binomial(1, reward_function_value[i])
        return reward_function_value, reward

    def best_arm(self, X):
        best_arms = np.zeros(X.shape[0], dtype = int)
        for i in range(X.shape[0]):
            best_arms[i] = self.func.best_arm(X[i])
        return best_arms
    

        