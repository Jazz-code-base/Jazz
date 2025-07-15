import numpy as np
import random
from typing import Optional, Tuple, Union, List

from dtqn.buffers.replay_buffer import ReplayBuffer
from dtqn.buffers.sum_tree import SumTree
from utils.bag import Bag

class PERBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay buffer, inherits from ReplayBuffer, uses SumTree for priority sampling
    
    Args:
        buffer_size: Buffer capacity
        env_obs_length: Environment observation vector length
        obs_mask: Padding value
        max_episode_steps: Maximum steps per episode
        context_len: Context length, default is 1
        alpha: Parameter determining the degree of prioritization, alpha=0 means uniform sampling, default is 0.6
        beta_start: Initial beta value for importance sampling weights, default is 0.4
        beta_frames: Number of frames for beta to grow from beta_start to 1.0, default is 100000
        eps: Small number added to TD error to ensure non-zero priority, default is 1e-6
    """
    def __init__(
        self,
        buffer_size: int,
        env_obs_length: Union[int, Tuple],
        obs_mask: int,
        max_episode_steps: int,
        context_len: Optional[int] = 1,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100000,
        eps: float = 1e-6
    ):
        super().__init__(buffer_size, env_obs_length, obs_mask, max_episode_steps, context_len)
        
        # PER parameters
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta_anneal_step = 0  # Current step for beta annealing, starts from 0
        self.eps = eps  # Small value added to TD error to ensure non-zero priority
        
        # Record maximum priority, used for new samples
        self.max_priority = 1.0
        
        # Each episode has an independent priority tree to track timestep-level priorities
        self.priority_trees = [SumTree(max_episode_steps) for _ in range(self.max_size)]
        
        # Flag indicating whether initial samples have been generated
        self.initial_samples_generated = False
    
    def get_beta(self) -> float:
        """
        Calculate current beta value for importance sampling weights
        """
        # Ensure beta_frames is not zero to avoid division by zero error
        if self.beta_frames == 0:
            return 1.0 # If frames is 0, beta is immediately 1
        return min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.beta_anneal_step / self.beta_frames))
    
    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,
    ) -> None:
        """
        Store transition and set maximum priority
        
        Args:
            obs: Observation
            action: Action
            reward: Reward
            done: Terminal flag
            episode_length: Current episode length (unused, kept for API compatibility)
        """
        # Get current episode index and timestep
        episode_idx = self.pos[0] % self.max_size
        obs_idx = self.pos[1]
        
        # Check if we've exceeded the buffer size
        if obs_idx + 1 > self.max_episode_steps:
            # If out of range, reset position and start a new episode
            self.flush()
            episode_idx = self.pos[0] % self.max_size
            obs_idx = 0
            self.cleanse_episode(episode_idx)
            # Clear the corresponding priority tree
            self.priority_trees[episode_idx] = SumTree(self.max_episode_steps)
            # Store observation at the beginning of the new episode
            self.obss[episode_idx, 0] = obs
            return
        
        # Call parent class storage logic
        super().store(obs, action, reward, done, episode_length)
        
        # Assign maximum priority to newly added transition
        self.priority_trees[episode_idx].add(self.max_priority ** self.alpha, obs_idx)
        
        # No longer incrementing beta annealing step here (self.frame_idx / self.beta_anneal_step)
    
    def sample(
        self, batch_size: int
    ) -> Tuple:
        """
        Sample batch_size transitions based on priority
        
        Returns:
            Sampled transition data: (obss, actions, rewards, next_obss, next_actions, dones, context_lengths,
                             weights, episode_indices_for_update, tree_node_indices_for_update, context_start_indices)
            Returns None if sampling fails.
        """
        if not self.can_sample(batch_size):
            return None

        valid_episodes_info = []  # (episode_idx, tree_total_priority, num_entries_in_tree)
        global_total_priority = 0.0
        total_valid_transitions = 0

        for i in range(min(self.pos[0], self.max_size)):
            if i == self.pos[0] % self.max_size: continue # Skip current filling episode
            if self.episode_lengths[i] > 0:
                tree = self.priority_trees[i]
                if tree.n_entries > 0 and tree.total() > 0:
                    valid_episodes_info.append((i, tree.total(), tree.n_entries))
                    global_total_priority += tree.total()
                    total_valid_transitions += tree.n_entries
        
        if not valid_episodes_info or global_total_priority <= 0:
            return None # Cannot sample

        sampled_episode_indices = []
        sampled_tree_node_indices = [] # For SumTree.update()
        sampled_priorities_p_i = []    # p_i of the specific transition
        context_start_indices = []     # Start index of the context window

        beta = self.get_beta()

        for _ in range(batch_size):
            rand_val = random.uniform(0, global_total_priority)
            current_sum = 0.0
            chosen_episode_idx_for_sample = -1
            
            for ep_idx, tree_total, _ in valid_episodes_info:
                if current_sum + tree_total >= rand_val:
                    chosen_episode_idx_for_sample = ep_idx
                    rand_val -= current_sum # Adjust rand_val to be local for this episode's tree
                    break
                current_sum += tree_total
            
            if chosen_episode_idx_for_sample == -1: # Fallback, should be rare
                chosen_episode_idx_for_sample, _, _ = random.choice(valid_episodes_info)
                rand_val = random.uniform(0, self.priority_trees[chosen_episode_idx_for_sample].total())

            tree = self.priority_trees[chosen_episode_idx_for_sample]
            tree_node_idx, priority_p_i, timestep_in_episode = tree.get(rand_val)
            
            sampled_episode_indices.append(chosen_episode_idx_for_sample)
            sampled_tree_node_indices.append(tree_node_idx)
            sampled_priorities_p_i.append(priority_p_i)

            # Determine context_start_index for the window that contains/starts at timestep_in_episode
            episode_len = self.episode_lengths[chosen_episode_idx_for_sample]
            if episode_len < self.context_len:
                start_idx = 0
            else:
                # Treat timestep_in_episode as the primary transition to build context around.
                # The context will be [start_idx, start_idx + context_len -1]
                # We want this primary transition to be included. A common way is to make it the last.
                # start_idx = max(0, timestep_in_episode - self.context_len + 1)
                # Or, if priority is for the *start* of a context, then timestep_in_episode *is* start_idx.
                # Given SumTree stores priority per individual timestep, let's assume `timestep_in_episode` is a point
                # and we build a context window that *starts* at or before it, and is valid.
                max_possible_start = episode_len - self.context_len
                start_idx = min(timestep_in_episode, max_possible_start)
                start_idx = max(0, start_idx)
            context_start_indices.append(start_idx)

        # Calculate Importance Sampling weights
        weights_np = np.zeros(batch_size, dtype=np.float32)
        if total_valid_transitions > 0:
            for i in range(batch_size):
                prob_i = sampled_priorities_p_i[i] / global_total_priority if global_total_priority > 0 else 0
                if prob_i > 0:
                    weights_np[i] = (total_valid_transitions * prob_i) ** (-beta)
                else: # if priority was 0 (e.g. from a fallback or just a 0 priority sample)
                    weights_np[i] = 0 # or some other indicator for zero probability samples
            
            if weights_np.max() > 0: # Normalize by max_weight
                weights_np = weights_np / weights_np.max()
            else: # All weights are zero, fallback to uniform (or could propagate error)
                 weights_np = np.ones(batch_size, dtype=np.float32)
        else: # No valid transitions, uniform weights (though sample should have returned None)
            weights_np = np.ones(batch_size, dtype=np.float32)

        # Fetch data for the sampled contexts
        transitions_list = []
        for i, start_idx_ctx in enumerate(context_start_indices):
            ep_len_ctx = self.episode_lengths[sampled_episode_indices[i]]
            if ep_len_ctx < self.context_len: # Episode shorter than context_len
                transitions_list.append(np.arange(self.context_len)) # Padded from start
            else:
                transitions_list.append(np.arange(start_idx_ctx, start_idx_ctx + self.context_len))
        
        transitions_np = np.array(transitions_list)
        formatted_episode_idxes_for_fetch = np.array([[idx] for idx in sampled_episode_indices])

        result_obss_list, result_actions_list, result_rewards_list = [], [], []
        result_next_obss_list, result_next_actions_list, result_dones_list = [], [], []

        for i in range(batch_size):
            ep_idx_fetch = formatted_episode_idxes_for_fetch[i, 0]
            trans_indices_ctx = transitions_np[i]
            
            valid_trans_ctx = np.clip(trans_indices_ctx, 0, self.max_episode_steps - 1)
            valid_next_trans_ctx = np.clip(trans_indices_ctx + 1, 0, self.max_episode_steps)

            result_obss_list.append(self.obss[ep_idx_fetch, valid_trans_ctx])
            result_actions_list.append(self.actions[ep_idx_fetch, valid_trans_ctx])
            result_rewards_list.append(self.rewards[ep_idx_fetch, valid_trans_ctx])
            result_next_obss_list.append(self.obss[ep_idx_fetch, valid_next_trans_ctx])
            result_next_actions_list.append(self.actions[ep_idx_fetch, valid_next_trans_ctx])
            result_dones_list.append(self.dones[ep_idx_fetch, valid_trans_ctx])

        sampled_ep_true_lengths = np.array([self.episode_lengths[idx] for idx in sampled_episode_indices])
        effective_context_lengths = np.clip(sampled_ep_true_lengths, 0, self.context_len)

        return (
            np.array(result_obss_list),
            np.array(result_actions_list),
            np.array(result_rewards_list),
            np.array(result_next_obss_list),
            np.array(result_next_actions_list),
            np.array(result_dones_list),
            effective_context_lengths, 
            weights_np,                     
            sampled_episode_indices,        
            sampled_tree_node_indices,      
            context_start_indices # Return context_start_indices for sample_with_bag
        )
    
    def sample_with_bag(
        self, batch_size: int, sample_bag: Bag
    ) -> Tuple:
        """
        Prioritized sampling with Bag mechanism
        """
        sample_results = self.sample(batch_size)
        
        if sample_results is None:
            return None
            
        (
            result_obss_np,
            result_actions_np,
            result_rewards_np,
            result_next_obss_np,
            result_next_actions_np,
            result_dones_np,
            effective_context_lengths,
            weights_np,
            sampled_episode_indices,
            sampled_tree_node_indices,
            sampled_context_start_indices # Newly added from self.sample()
        ) = sample_results
        
        bag_obss_np = np.full(
            [batch_size, sample_bag.size, sample_bag.obs_length],
            sample_bag.obs_mask, dtype=self.obss.dtype # Match dtype
        )
        bag_actions_np = np.full(
            [batch_size, sample_bag.size, 1],
            0, dtype=self.actions.dtype # Match dtype
        )
        
        for i in range(batch_size):
            ep_idx = sampled_episode_indices[i]
            # Use the context start for bag logic, similar to original
            start_for_bag_logic = sampled_context_start_indices[i]
            episode_len_for_ep = self.episode_lengths[ep_idx]

            # Original bag sampling logic based on `start_for_bag_logic` (context start)
            if start_for_bag_logic < sample_bag.size:
                # Take all elements before the context start, up to sample_bag.size
                num_to_take = min(start_for_bag_logic, episode_len_for_ep, sample_bag.size)
                if num_to_take > 0:
                    bag_obss_np[i, :num_to_take] = self.obss[ep_idx, :num_to_take]
                    bag_actions_np[i, :num_to_take] = self.actions[ep_idx, :num_to_take]
            else:
                # Sample randomly from elements before the context start
                valid_range_for_bag = min(start_for_bag_logic, episode_len_for_ep)
                if valid_range_for_bag > 0:
                    num_to_sample_for_bag = min(sample_bag.size, valid_range_for_bag)
                    
                    indices_to_sample_from = np.arange(valid_range_for_bag)
                    sampled_bag_item_indices = random.sample(list(indices_to_sample_from), num_to_sample_for_bag)
                    
                    for k, bag_item_idx in enumerate(sampled_bag_item_indices):
                        bag_obss_np[i, k] = self.obss[ep_idx, bag_item_idx]
                        bag_actions_np[i, k, 0] = self.actions[ep_idx, bag_item_idx, 0]
        
        return (
            result_obss_np,
            result_actions_np,
            result_rewards_np,
            result_next_obss_np,
            result_next_actions_np,
            result_dones_np,
            effective_context_lengths,
            weights_np,
            sampled_episode_indices,
            sampled_tree_node_indices,
            # sampled_context_start_indices, # Not explicitly returned by sample_with_bag to agent usually
            bag_obss_np,
            bag_actions_np,
        )
    
    def update_priorities(
        self,
        sampled_episode_indices: List[int],
        sampled_tree_node_indices: List[int],
        td_errors: np.ndarray, # Expected shape [batch_size] or [batch_size, 1]
    ) -> None:
        """
        Update priorities using newly calculated TD errors.
        td_errors (np.ndarray): TD errors for each sampled transition, should be 1D array [batch_size].
                                 If agent sends [batch_size, context_len], caller needs to aggregate.
        """
        # Ensure td_errors is effectively [batch_size]
        if len(td_errors.shape) > 1:
             # If agent sends [batch_size, context_len], we take the mean along context_len axis.
             # This assumes the priority corresponds to the average error of the context, 
             # or that the DtqnAgent has already processed it to be representative.
            effective_td_errors = np.mean(td_errors, axis=1)
        else:
            effective_td_errors = td_errors.flatten()

        if not (len(sampled_episode_indices) == len(sampled_tree_node_indices) == len(effective_td_errors)):
            print(f"Warning: update_priorities dimension mismatch. Ep: {len(sampled_episode_indices)}, Node: {len(sampled_tree_node_indices)}, TD: {len(effective_td_errors)}")
            return

        for i in range(len(sampled_episode_indices)):
            episode_idx = sampled_episode_indices[i]
            tree_node_idx = sampled_tree_node_indices[i] 
            td_error_for_sample = effective_td_errors[i]
            
            if not (0 <= episode_idx < len(self.priority_trees)):
                print(f"Warning: update_priorities invalid episode_idx {episode_idx}")
                continue
                
            tree = self.priority_trees[episode_idx]
            new_priority_val = (abs(td_error_for_sample) + self.eps) ** self.alpha
            tree.update(tree_node_idx, new_priority_val)
            self.max_priority = max(self.max_priority, new_priority_val) # Update global max_priority
    
    def step_beta_annealing(self):
        """Called after each training step to advance the beta annealing process."""
        if self.beta_anneal_step < self.beta_frames:
            self.beta_anneal_step += 1

    def total_priority(self) -> float:
        """
        Calculate the sum of all priority trees
        
        Returns:
            Sum of all priorities
        """
        # Exclude the current episode being filled
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size and self.episode_lengths[i] > 0
        ]
        
        # Calculate the sum of all valid priority trees
        total = 0.0
        for idx in valid_episodes:
            total += self.priority_trees[idx].total()
            
        return total if total > 0 else 1.0  # Avoid division by zero
    
    def can_sample(self, batch_size: int) -> bool:
        """
        Check if there are enough samples for sampling
        
        Args:
            batch_size: Batch size
            
        Returns:
            True if there are enough valid episodes, False otherwise
        """
        # Get the number of valid episodes
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size and self.episode_lengths[i] > 0
        ]
        
        # Check if there are enough valid episodes
        return len(valid_episodes) >= 1  # Only need one valid episode, as can sample multiple times from the same episode
    
    def flush(self) -> None:
        """
        Called when the current episode ends, updates state
        """
        # Call parent class flush method
        super().flush()
        
        # No additional cleanup needed here, as we create new SumTree in store

    def cleanse_episode(self, episode_idx: int) -> None:
        """
        Clear data for the specified episode and reset its SumTree
        """
        super().cleanse_episode(episode_idx)
        # Reset the SumTree associated with this episode slot
        if 0 <= episode_idx < len(self.priority_trees):
            self.priority_trees[episode_idx] = SumTree(self.max_episode_steps)
            # print(f"DEBUG: Cleansed SumTree for episode_idx {episode_idx}") # Debug log
        else:
            # This should not happen, but as a safeguard
            print(f"Warning: cleanse_episode attempted to access invalid episode_idx {episode_idx} to reset SumTree") 