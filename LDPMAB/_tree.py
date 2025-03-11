import numpy as np
from collections import deque
from copy import deepcopy
import time

_TREE_LEAF = -1
_TREE_UNDEFINED = -2




class TreeStruct(object):
    """ Basic Binary Tree Structure.
    """
    def __init__(self, 
                 splitter, 
                 estimator, 
                 max_depth,
                 epsilon_target,
                 epsilon_list,
                 n_target,
                 n_source_list,
                 n_max,
                 K,
                 M,
                 dim,
                 X_range,
                 batch_size,
                 warm_start_len,
                 ):
        
        self.splitter = splitter()
        self.estimator = estimator
        self.max_depth = max_depth
        self.epsilon_list = epsilon_list
        self.n_target = n_target
        self.n_source_list = n_source_list
        self.n_max = n_max
        self.epsilon_target = epsilon_target
        self.K = K
        self.dim = dim
        self.X_range = X_range
        self.M = M
        self.batch_size = batch_size
        self.warm_start_len = warm_start_len

        self.active_depth = 0
        self.node_count = 0
        self.left_child = []
        self.right_child = []
        self.feature = []
        self.threshold = []
        self.leaf_ids = []
        self.leafnode_fun = {} 
        self.allnode_fun = {}
        self.parent = []
        self.node_range = []

    def _node_append(self):
        """Add None to each logs as placeholders. 
        """
        self.left_child.append(None)
        self.right_child.append(None)
        self.feature.append(None)
        self.threshold.append(None)
        self.node_range.append(None)  
            
    def _add_node(self, parent, is_left, is_leaf, feature, threshold, node_range=None):
        """Add a new node. 
        """
        
        self.parent.append(parent)
        self._node_append()
        node_id = self.node_count
        self.node_range[node_id] = node_range.copy()
            
        # record children status in parent nodes
        if parent != _TREE_UNDEFINED:
            if is_left:
                self.left_child[parent] = node_id
            else:
                self.right_child[parent] = node_id
        # record current node status
        if is_leaf:
            self.left_child[node_id] = _TREE_LEAF
            self.right_child[node_id] = _TREE_LEAF
            self.feature[node_id] = _TREE_UNDEFINED
            self.threshold[node_id] = _TREE_UNDEFINED
            self.leaf_ids.append(node_id)  
        else:
            # left_child and right_child will be set later
            self.feature[node_id] = feature
            self.threshold[node_id] = threshold
        self.node_count += 1
        return node_id

 
    
    def _node_info_to_ndarray(self):
        """Turn each logs into arrays. 
        """
        self.left_child = np.array(self.left_child, dtype=np.int32)
        self.right_child = np.array(self.right_child, dtype=np.int32)
        self.feature = np.array(self.feature, dtype=np.int32)
        self.threshold = np.array(self.threshold, dtype=np.float64)
        self.leaf_ids = np.array(self.leaf_ids, dtype=np.int32)
        self.node_range = np.array(self.node_range, dtype=np.float64)
            
    def apply(self, X):
        
        """Get node ids.
        """
        n = X.shape[0]
        result_nodeid = np.zeros(n, dtype=np.int32)
        for i in range(n):
            node_id = 0
            while self.left_child[node_id] != _TREE_LEAF:
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
            result_nodeid[i] = node_id
        return  result_nodeid  
    
    def apply_active(self, X):
        
        """Get node ids.
        """
        n = X.shape[0]
        result_nodeid = np.zeros(n, dtype=np.int32)
        for i in range(n):
            node_id = 0
            while self.left_child[node_id] != _TREE_LEAF and not self.leafnode_fun[node_id].active_flag:
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
            result_nodeid[i] = node_id
        return  result_nodeid  
    
    def get_node_range(self, node_id):
        """Get the range of the node. 
        """
        return self.node_range[node_id]
    
    def get_all_anscestor(self, X):
        n = X.shape[0]
        result_nodeid = np.zeros((n, self.max_depth + 1), dtype=np.int32)
        for i in range(n):
            node_id = 0
            for s in range(self.max_depth + 1):
                result_nodeid[i, s] = node_id
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
        return  result_nodeid  

    def get_all_descendant_leaf(self, node_idx):
        
        child_idx_list = deque()
        child_idx_list.append(node_idx)
        final_idx_list = []
        
        while len(child_idx_list) != 0:
            current_idx = child_idx_list.popleft()
            if self.left_child[current_idx] != _TREE_LEAF:
                child_idx_list.append(self.left_child[current_idx]) 
            if self.right_child[current_idx] != _TREE_LEAF:
                child_idx_list.append(self.right_child[current_idx])
            if self.left_child[current_idx] == _TREE_LEAF and self.right_child[current_idx] == _TREE_LEAF:
                final_idx_list.append(current_idx)
        
        return final_idx_list

    def get_all_descendant(self, node_idx):
        descendant_idx = []
        descendant_idx.append(node_idx)
        stack = deque()
        stack.append(node_idx)
        while len(stack) != 0:
            current_idx = stack.popleft()
            if self.left_child[current_idx] != _TREE_LEAF:
                descendant_idx.append(self.left_child[current_idx])
                stack.append(self.left_child[current_idx])
            if self.right_child[current_idx] != _TREE_LEAF:
                descendant_idx.append(self.right_child[current_idx])
                stack.append(self.right_child[current_idx])
        return descendant_idx

    def get_active_arms(self, X):
        """Get active arms.
        """
        leaf_nodes = self.apply_active(X)
        active_arms = []
        for leaf_node in leaf_nodes:
            active_arms.append(self.leafnode_fun[leaf_node].active_arms)
        return active_arms

    def policy(self, X):
        """Get policy.
        """
        active_arms = self.get_active_arms(X)
        return [np.random.choice(active_arms[i]) for i in range(len(active_arms))]
    
    
    def get_estimation(self, X):
        arm_scores = []
        active_arms = self.get_active_arms(X)
        n = X.shape[0]
        for i in range(n):
            node_id = 0
            while self.left_child[node_id] != _TREE_LEAF:
                if X[i, self.feature[node_id]] < self.threshold[node_id]:
                    node_id = self.left_child[node_id]
                else:
                    node_id = self.right_child[node_id]
                if self.leafnode_fun[node_id].active_flag:
                    break
                
            arm_scores.append([self.leafnode_fun[node_id].estimation[i] for i in active_arms[i]])
        return active_arms, arm_scores

    def exponential_weight(self, score, K):
        score = np.array(score)
        if np.max(score) - np.min(score) < 0.01:
            return np.ones(score.shape[0]) / score.shape[0]
        else:
            score = np.exp(score * np.log(1.2 * K / score.shape[0])  / (np.max(score) - np.min(score) + 1e-4))
            
            return score / score.sum()


    def weighted_policy(self, X):
        active_arms, arm_scores  = self.get_estimation(X)
        return [np.random.choice(active_arms[i], p = self.exponential_weight(arm_scores[i], self.K)) for i in range(len(active_arms))]

    def remove_descendent_active_arms(self, node_idx, k):
        """Remove all active arms of the descendent nodes of node_idx.
        """
        removal_node_active_arms = self.leafnode_fun[node_idx].active_arms.copy()
        descendant_idx = self.get_all_descendant(node_idx)
        for idx in descendant_idx:
            self.leafnode_fun[idx].active_arms = np.delete(self.leafnode_fun[idx].active_arms, np.where(self.leafnode_fun[idx].active_arms == k))
            if self.leafnode_fun[idx].active_arms.shape[0] == 0:
                raise ValueError("active arms is empty, remove id: {}, node_id: {}, active_arms: {}, removal_node_active_arms: {}".format(node_idx, idx, self.leafnode_fun[idx].active_arms, removal_node_active_arms))
                self.leafnode_fun[idx].active_arms = removal_node_active_arms

    ###################################################################################################################################################

   
                
                   
    
    def visualize_node(self, node_id):
        print("id: ", node_id)
        print("range: ", self.get_node_range(node_id))
        print("depth: ", self.leafnode_fun[node_id].node_depth)
        print("active_arms: ", self.leafnode_fun[node_id].active_arms)
        print("active_flag: ", self.leafnode_fun[node_id].active_flag)
        print("tau: ", self.leafnode_fun[node_id].tau)
        print("estimations: ", self.leafnode_fun[node_id].estimation)
        print("r: ", self.leafnode_fun[node_id].r)
        

    def visualize_path(self, x):
        ancestors_id = self.get_all_anscestor(x).ravel()
        former_id = 0
        for ancestor_id in ancestors_id:
            if not self.leafnode_fun[former_id].active_flag:
                self.visualize_node(ancestor_id)
            former_id = ancestor_id

    def visualize_estimator(self, x, k):
        print('{0:*^40}'.format("visualizing the estimator"))
        ancestors_id = self.get_all_anscestor(x).ravel()
        node_id = ancestors_id[self.active_depth]
        print('{0:*^40}'.format("arm {}, node range: {}, step :{}".format(k, self.leafnode_fun[node_id].X_range, self.leafnode_fun[node_id].t_target)))
        self.leafnode_fun[node_id].visualize_estimator(k)


    
    def initialize(self):
        """Grow the tree.
        """
        stack = deque()
        stack.append([self.X_range, _TREE_UNDEFINED, _TREE_UNDEFINED, 0])
        while len(stack) != 0:

            
            node_range, parent, is_left, depth = stack.popleft()

            if depth >= self.max_depth:
                is_leaf = True
            else: 
                is_leaf = False

            rd_dim, rd_split = self.splitter(node_range)                
            if not is_leaf:
                node_id = self._add_node(parent, is_left, is_leaf, rd_dim, rd_split, node_range)
            else:
                node_id = self._add_node(parent, is_left, is_leaf, None, None, node_range)
  
            self.leafnode_fun[node_id] = self.estimator( 
                                                        self.epsilon_target,
                                                        self.epsilon_list,
                                                        node_range,
                                                        self.M,
                                                        self.K,
                                                        self.n_max,
                                                        self.dim,
                                                        self.max_depth,
                                                        depth,
                                                        self.batch_size,
                                                        self.warm_start_len,
                                                        )
            if not is_leaf:
                if node_range is not None:
                    node_range_right = node_range.copy()
                    node_range_left = node_range.copy()
                    node_range_right[0, rd_dim] = rd_split
                    node_range_left[1, rd_dim] = rd_split
                else:
                    node_range_right = node_range_left = None

                stack.append([node_range_right, node_id, False, depth + 1])
                stack.append([node_range_left, node_id, True, depth + 1])
            
        self._node_info_to_ndarray()
  
    def apply_update(self, X, arms, y, source_id):
        stack = deque()
        stack.append([X, arms, y, 0, 0])
        if_push_active_depth = True 
        
        while len(stack) != 0:
            
            dt_X, dt_arms, dt_y, node_id, depth = stack.popleft()

            if depth <= self.max_depth:
                if_updated, removed_arms = self.leafnode_fun[node_id].update(dt_X, dt_arms, dt_y, source_id)
                if depth == self.active_depth:
                    if self.leafnode_fun[node_id].active_flag:
                        if_push_active_depth = False
                
                for k in removed_arms:
                    self.remove_descendent_active_arms(node_id, k)
                
                if if_updated:
                    continue

                if dt_X.shape[0] == 0:
                    dt_X_left = dt_X
                    dt_arms_left = dt_arms
                    dt_y_left = dt_y

                    dt_X_right = dt_X
                    dt_arms_right = dt_arms
                    dt_y_right = dt_y

                else:
                    right_idx = dt_X[:, self.feature[node_id]] >= self.threshold[node_id]
                    left_idx = ~right_idx

                    dt_X_left = dt_X[left_idx]
                    dt_arms_left = dt_arms[left_idx]
                    dt_y_left = dt_y[left_idx]
                    dt_X_right = dt_X[right_idx]
                    dt_arms_right = dt_arms[right_idx]
                    dt_y_right = dt_y[right_idx]

                stack.append([dt_X_right, dt_arms_right, dt_y_right, self.right_child[node_id], depth + 1])
                stack.append([dt_X_left, dt_arms_left, dt_y_left, self.left_child[node_id], depth + 1])
               
        if if_push_active_depth:
            self.active_depth += 1
       