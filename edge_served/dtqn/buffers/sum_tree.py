import numpy as np

class SumTree:
    """
    SumTree data structure for prioritized experience replay
    """
    def __init__(self, capacity):
        """
        Initialize SumTree
        
        Args:
            capacity: Capacity of leaf nodes in the tree
        """
        # Number of tree nodes equals 2 * capacity - 1, where capacity is the number of leaf nodes
        self.capacity = capacity
        # Initialize tree with internal nodes + leaf nodes
        self.tree = np.zeros(2 * capacity - 1)
        # Initialize data storage
        self.data = np.zeros(capacity, dtype=object)
        # Track number of entries added
        self.n_entries = 0
        # Points to the next write position
        self.write = 0
    
    def _propagate(self, idx, change):
        """
        Update the sum of all parent nodes
        
        Args:
            idx: Index of the changed node
            change: Value of the change
        """
        parent = (idx - 1) // 2
        self.tree[parent] += change
        
        # If not root node, continue propagating upward
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx, s):
        """
        Find the sample index based on priority value s in the tree
        
        Args:
            idx: Current node index
            s: Priority sum value
            
        Returns:
            Corresponding sample index
        """
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
            
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self):
        """
        Return the total sum of the tree
        """
        return self.tree[0]
    
    def add(self, p, data):
        """
        Add new sample with priority
        
        Args:
            p: Priority value
            data: Sample data (can be an index)
        """
        # Find leaf node index
        idx = self.write + self.capacity - 1
        
        # Store data
        self.data[self.write] = data
        
        # Update tree
        self.update(idx, p)
        
        # Update write position
        self.write = (self.write + 1) % self.capacity
        
        # Update entry count
        if self.n_entries < self.capacity:
            self.n_entries += 1
    
    def update(self, idx, p):
        """
        Update priority
        
        Args:
            idx: Index of node to update
            p: New priority value
        """
        # Calculate change amount
        change = p - self.tree[idx]
        
        # Update node
        self.tree[idx] = p
        
        # Propagate change upward
        self._propagate(idx, change)
    
    def get(self, s):
        """
        Get sample based on priority sum s
        
        Args:
            s: Value between 0 and total sum
            
        Returns:
            (idx, priority, data): Tree index, priority, sample data
        """
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        return (idx, self.tree[idx], self.data[dataIdx]) 