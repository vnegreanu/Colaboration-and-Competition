import numpy as np
from utils.sum_tree import SumTree

class PER():  # stored as ( s, a, r, s_new, done ) in SumTree
    """
    This PER code is modified version of the code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0..1] convert the importance of TD error to priority, often 0.6
    beta = 0.4  # importance-sampling, from initial value increasing to 1, often 0.4
    beta_increment_per_sampling = 1e-4  # annealing the bias, often 1e-3
    absolute_error_upper = 1.   # clipped abs error
    
    def __init__(self, capacity):
        """
        The tree is composed of a sum tree that contains the priority scores at his leaf and also a data array.
        """
        self.tree = SumTree(capacity)
    
    def __len__(self):
        return len(self.tree)
    
    def is_full(self):
        return len(self.tree) >= self.tree.capacity
    
    def add(self, sample, error = None):
        if error is None:
            priority = np.amax(self.tree.tree[-self.tree.capacity:])
            if priority == 0: priority = self.absolute_error_upper
        else:
            priority = min((abs(error) + self.epsilon) ** self.alpha, self.absolute_error_upper)
        self.tree.add(sample, priority)
    
    def sample(self, n):
        """
        - First, to sample a minibatch of size k the range [0, priority_total] is divided into k ranges.
        - Then a value is uniformly sampled from each range.
        - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
        - Then, we calculate IS weights for each minibatch element.
        """
        
        minibatch = []
        
        idxs = np.empty((n,), dtype=np.int32)
        is_weights = np.empty((n,), dtype=np.float32)
        
        # Calculate the priority segment
        # Divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n       # priority segment
        
        # Increase the beta each time we sample a new minibatch
        self.beta = np.amin([1., self.beta + self.beta_increment_per_sampling])  # max = 1
        
        # Calculate the max_weight
        p_min = np.amin(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.beta)
        
        for i in range(n):
            """
            A value is uniformly sampled from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)
            
            """
            Experience that corresponds to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)
            
            sampling_probabilities = priority / self.tree.total_priority
            is_weights[i] = np.power(n * sampling_probabilities, -self.beta)/ max_weight
            
            idxs[i]= index
            minibatch.append(data)
            
        return idxs, minibatch, is_weights
    
    def batch_update(self, idxs, errors):
        """
        Update the priorities on the tree
        """
        errors = errors + self.epsilon
        clipped_errors = np.minimum(errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.alpha)
        
        for idx, p in zip(idxs, ps):
            self.tree.update(idx, p)