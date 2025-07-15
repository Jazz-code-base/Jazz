import numpy as np
import random
from typing import Optional, Tuple, Union

from utils.bag import Bag


class ReplayBuffer:
    """
    FIFO Replay Buffer which stores contexts of length ``context_len`` rather than single
        transitions

    Args:
        buffer_size: The number of transitions to store in the replay buffer
        env_obs_length: The size (length) of the environment's observation
        context_len: The number of transitions that will be stored as an agent's context. Default: 1
    """

    def __init__(
        self,
        buffer_size: int,
        env_obs_length: Union[int, Tuple],
        obs_mask: int,
        max_episode_steps: int,
        context_len: Optional[int] = 1,
    ):
        self.max_size = buffer_size // max_episode_steps
        self.context_len = context_len
        self.env_obs_length = env_obs_length
        self.max_episode_steps = max_episode_steps
        self.obs_mask = obs_mask
        self.pos = [0, 0] # [episode count, number of transitions added in current episode]

        # Image domains
        if isinstance(env_obs_length, tuple):
            self.obss = np.full(
                [
                    self.max_size,
                    max_episode_steps + 1,  # Keeps first and last obs together for +1
                    *env_obs_length,
                ],
                obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obss = np.full(
                [
                    self.max_size,
                    max_episode_steps + 1,  # Keeps first and last obs together for +1
                    env_obs_length,
                ],
                obs_mask,
                dtype=np.float32,
            )

        # Need the +1 so we have space to roll for the first observation
        self.actions = np.zeros(
            [self.max_size, max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards = np.zeros(
            [self.max_size, max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones = np.ones(
            [self.max_size, max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths = np.zeros([self.max_size], dtype=np.uint32)

    def store(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        episode_length: Optional[int] = 0,  # This parameter is now unused but kept for API consistency
    ) -> None:
        episode_idx = self.pos[0] % self.max_size
        obs_idx = self.pos[1]
        
        # Check if we've exceeded the buffer size
        if obs_idx + 1 > self.max_episode_steps: # Simplified condition: obs_idx can go from 0 to max_episode_steps-1 for actions/rewards
            # If out of range, reset position and start a new episode
            self.flush()
            episode_idx = self.pos[0] % self.max_size
            obs_idx = 0 # obs_idx is reset by flush through self.pos[1]
            self.cleanse_episode(episode_idx)
            # Store observation at the beginning of the new episode
            self.obss[episode_idx, 0] = obs # This is s_0 for the new episode
            # The corresponding action, reward, done for s_0 will be stored in the next call to store()
            # No need to store action/reward/done here as this obs is the *first* of a new episode.
            return
            
        self.obss[episode_idx, obs_idx + 1] = obs
        self.actions[episode_idx, obs_idx] = action
        self.rewards[episode_idx, obs_idx] = reward
        self.dones[episode_idx, obs_idx] = done
        # self.episode_lengths[episode_idx] = episode_length # Removed: episode_lengths updated in flush
        self.pos = [self.pos[0], self.pos[1] + 1]

    def store_first_obs(self, obs: np.ndarray) -> None:
        """Use this at the beginning of the episode to store the first obs"""
        episode_idx = self.pos[0] % self.max_size
        self.cleanse_episode(episode_idx)
        self.obss[episode_idx, 0] = obs

    def can_sample(self, batch_size: int) -> bool:
        # Simple check: whether there are enough episodes
        valid_episodes_count = min(self.pos[0], self.max_size)
        # Print simple debug info
        # print(f"Buffer sampling check - batch_size: {batch_size}, valid episodes: {valid_episodes_count}")
        if batch_size > valid_episodes_count:
            # print(f"Buffer sampling failed: valid episode count ({valid_episodes_count}) less than batch_size ({batch_size})")
            return False
        return True

    def flush(self):
        # This method is called when a logical episode ends and is explicitly flushed.
        # Before switching to the next buffer episode slot,
        # record the actual length of the completed logical episode in the current slot.
        current_episode_index = self.pos[0] % self.max_size
        if self.pos[1] > 0 : # Ensure data was actually stored in this slot
            self.episode_lengths[current_episode_index] = self.pos[1] # Record actual transitions stored

        # Switch to next buffer episode slot and reset in-slot counter
        self.pos = [self.pos[0] + 1, 0]

    def cleanse_episode(self, episode_idx: int) -> None:
        # Cleanse the episode of any previous data
        # Image domains
        if isinstance(self.env_obs_length, tuple):
            self.obss[episode_idx] = np.full(
                [
                    self.max_episode_steps
                    + 1,  # Keeps first and last obs together for +1
                    *self.env_obs_length,
                ],
                self.obs_mask,
                dtype=np.uint8,
            )
        else:
            self.obss[episode_idx] = np.full(
                [
                    self.max_episode_steps
                    + 1,  # Keeps first and last obs together for +1
                    self.env_obs_length,
                ],
                self.obs_mask,
                dtype=np.float32,
            )
        self.actions[episode_idx] = np.zeros(
            [self.max_episode_steps + 1, 1],
            dtype=np.uint8,
        )
        self.rewards[episode_idx] = np.zeros(
            [self.max_episode_steps, 1],
            dtype=np.float32,
        )
        self.dones[episode_idx] = np.ones(
            [self.max_episode_steps, 1],
            dtype=np.bool_,
        )
        self.episode_lengths[episode_idx] = 0 # Reset length for the new clean episode

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # Exclude the episode currently being filled
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size
        ]
        
        if len(valid_episodes) == 0:
            print(f"Warning: No valid episodes available for sampling")
            # If no episodes are available, create an empty episode index
            valid_episodes = [0]  # Use the first episode (even if it's currently being filled)
        
        # Randomly select from valid episodes
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
        )
        
        # Handle transition_starts, ensure they don't go out of bounds
        transition_starts = []
        for idx_arr in episode_idxes: # Renamed idx to idx_arr to avoid conflict
            episode_idx = idx_arr[0]
            # episode_len now represents the actual number of transitions stored in this episode slot
            episode_len = self.episode_lengths[episode_idx]
            
            # If episode length is 0 (e.g., never properly filled or cleared and not filled),
            # or less than the context_len needed for sampling, we need special handling to avoid errors.
            if episode_len == 0: # Should not happen if can_sample ensures valid episodes are chosen
                # This case indicates a problem, possibly sampling an empty/cleansed episode.
                # For robustness, treat as if episode_len is min_context_len for sampling, though data will be padding.
                # However, valid_episodes should not include empty ones if logic is correct.
                # Defaulting to start=0 and minimal length for safety, but this signals an issue.
                print(f"Warning: Sampled episode {episode_idx} with length 0. This should not happen.")
                transition_starts.append(0)
                # This will likely lead to sampling padding values or an error if not handled by caller.
            elif episode_len < self.context_len:
                transition_starts.append(0)  # Sample from the beginning, subsequent handling will be padding
            else:
                # Randomly select a starting position to ensure it doesn't go out of bounds
                max_start = episode_len - self.context_len
                transition_starts.append(random.randint(0, max_start))
        
        transition_starts = np.array(transition_starts)
        
        # Create a transition sequence for each sampled transition
        transitions_list = []
        for i, start in enumerate(transition_starts):
            episode_idx = episode_idxes[i][0]
            episode_len = self.episode_lengths[episode_idx]
            
            # If episode length is insufficient for context_len, need special handling
            # Or if episode_len is 0 (already handled above, but double-check for safety)
            if episode_len == 0:
                 # Create an empty range of length context_len, if episode_len is 0
                 # This will result in sampling padding values (obs_mask)
                 # Or, more upstream logic needs to handle this case, e.g., not selecting such episodes
                 transitions_list.append(np.arange(self.context_len)) # Sample of context_len, will pick padding
            elif episode_len < self.context_len:
                # Use actual length, avoid overflow, but will pad to context_len
                # The sampled sequence will be shorter than context_len, padding is handled by network
                # Or, we can create a sequence of context_len starting at 0, and rely on obs_mask for padding
                # For DTQN, fixed context length is expected.
                # Create a sequence of context_len, actual data up to episode_len, rest will be padding.
                # np.arange(start, start + self.context_len) would be [0, context_len-1]
                # We need to ensure indices are within `episode_len`.
                # The actual data is from 0 to episode_len-1. Sample [0, context_len-1]
                # and rely on data being obs_mask beyond episode_len.
                # This means self.obss should correctly have obs_mask for unwritten parts.
                transitions_list.append(np.arange(self.context_len)) # Sample indices [0, ..., context_len-1]
            else:
                # Use normal starting position to starting position + context_len
                transitions_list.append(np.arange(start, start + self.context_len))
                
        # # Debug print start
        # print("\n" + "="*40)
        # print(f"Sampling check - Total samples: {len(transitions_list)}")
        # for idx, trans in enumerate(transitions_list):
        #     print(f"Sample {idx} Length: {len(trans)} | Content: {trans[:5]}...")  # Show first 5 elements
        # print("="*40 + "\n")
        # # Debug print end
        
        transitions = np.array(transitions_list)
        
        # Handle potential index overflow issues
        result_obss = []
        result_actions = []
        result_rewards = []
        result_next_obss = []
        result_next_actions = []
        result_dones = []
        
        for i in range(batch_size):
            episode_idx = episode_idxes[i][0]
            trans = transitions[i]
            
            # Ensure all indices are within valid range
            valid_trans = np.clip(trans, 0, self.max_episode_steps - 1)
            valid_next_trans = np.clip(trans + 1, 0, self.max_episode_steps)
            
            # Collect data for the current batch
            result_obss.append(self.obss[episode_idx, valid_trans])
            result_actions.append(self.actions[episode_idx, valid_trans])
            result_rewards.append(self.rewards[episode_idx, valid_trans])
            result_next_obss.append(self.obss[episode_idx, valid_next_trans])
            result_next_actions.append(self.actions[episode_idx, valid_next_trans])
            result_dones.append(self.dones[episode_idx, valid_trans])
        
        # Convert results to numpy arrays
        result_obss = np.array(result_obss)
        result_actions = np.array(result_actions)
        result_rewards = np.array(result_rewards)
        result_next_obss = np.array(result_next_obss)
        result_next_actions = np.array(result_next_actions)
        result_dones = np.array(result_dones)
        
        return (
            result_obss,
            result_actions,
            result_rewards,
            result_next_obss,
            result_next_actions,
            result_dones,
            np.clip(self.episode_lengths[episode_idxes], 0, self.context_len).reshape(-1),
        )

    def sample_with_bag(
        self, batch_size: int, sample_bag: Bag
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        # Select valid episodes
        valid_episodes = [
            i
            for i in range(min(self.pos[0], self.max_size))
            if i != self.pos[0] % self.max_size
        ]
        
        if len(valid_episodes) == 0:
            print(f"Warning: No valid episodes available for bag sampling")
            # If no episodes are available, create an empty episode index
            valid_episodes = [0]  # Use the first episode (even if it's currently being filled)
        
        # Randomly select from valid episodes
        episode_idxes = np.array(
            [[random.choice(valid_episodes)] for _ in range(batch_size)]
        )
        
        # Before subtracting context_len, ensure the result is positive
        # Here we need to ensure the calculation result is valid, minimum is 0, and does not exceed the actual length of the episode
        transition_starts = []
        for idx_arr in episode_idxes: # Renamed idx to idx_arr to avoid conflict
            episode_idx = idx_arr[0]
            episode_len = self.episode_lengths[episode_idx]
            
            # Ensure at least context_len, otherwise use padding
            if episode_len == 0:
                print(f"Warning: Bag sampled episode {episode_idx} with length 0.")
                transition_starts.append(0)
            elif episode_len < self.context_len:
                transition_starts.append(0)  # Sample from the beginning, subsequent handling will be padding
            else:
                # Randomly select a starting position to ensure it doesn't go out of bounds
                # max_start = episode_len - self.context_len # Note: Here episode_len is context.timestep, which is the global time step in the training process, not the length of this episode
                max_start = episode_len - self.context_len
                transition_starts.append(random.randint(0, max_start))
        
        transition_starts = np.array(transition_starts)
        
        # Create a transition sequence for each sampled transition
        # Note: Here we need to check if the length of each episode is sufficient
        transitions_list = []
        for i, start in enumerate(transition_starts):
            episode_idx = episode_idxes[i][0]
            episode_len = self.episode_lengths[episode_idx]
            
            # If episode length is insufficient for context_len, need special handling
            if episode_len == 0:
                transitions_list.append(np.arange(self.context_len))
            elif episode_len < self.context_len:
                # Use actual length, avoid overflow
                # transitions_list.append(np.arange(min(episode_len, self.context_len)))
                transitions_list.append(np.arange(self.context_len)) # Sample full context, rely on padding
            else:
                # Use normal starting position to starting position + context_len
                transitions_list.append(np.arange(start, start + self.context_len))
                
        # # Debug print start
        # print("\n" + "="*40)
        # print(f"Sampling check - Total samples: {len(transitions_list)}")
        # for idx, trans in enumerate(transitions_list):
        #     print(f"Sample {idx} Length: {len(trans)} | Content: {trans[:5]}...")  # Show first 5 elements
        # print("="*40 + "\n")
        # # Debug print end
        
        transitions = np.array(transitions_list)

        # Create batch_size copies of bags
        bag_obss = np.full(
            [batch_size, sample_bag.size, sample_bag.obs_length],
            sample_bag.obs_mask,
        )
        bag_actions = np.full(
            [batch_size, sample_bag.size, 1],
            0,
        )

        # Sample observations from observations not in the main context into the bag
        for bag_idx in range(batch_size):
            episode_idx = episode_idxes[bag_idx][0]
            start = transition_starts[bag_idx]
            episode_len = self.episode_lengths[episode_idx]
            
            # Possible bag is smaller than max bag size, so take all of it
            if start < sample_bag.size:
                valid_start = min(start, episode_len)  # Ensure it does not exceed episode length
                bag_obss[bag_idx, :valid_start] = self.obss[
                    episode_idxes[bag_idx], :valid_start
                ]
                bag_actions[bag_idx, :valid_start] = self.actions[
                    episode_idxes[bag_idx], :valid_start
                ]
            # Otherwise, randomly sample
            else:
                # Ensure the random sampling range does not exceed episode length
                valid_range = min(start, episode_len)
                if valid_range > 0:  # Only sample if there is valid data
                    sample_size = min(sample_bag.size, valid_range)  # Ensure it does not exceed available data
                    
                    # Convert data to a list for sampling
                    obs_list = self.obss[episode_idxes[bag_idx], :valid_range].reshape(-1, sample_bag.obs_length).tolist()
                    action_list = self.actions[episode_idxes[bag_idx], :valid_range].reshape(-1).tolist()
                    
                    # Randomly sample, ensuring it does not exceed available data
                    sampled_indices = random.sample(range(len(obs_list)), min(sample_size, len(obs_list)))
                    
                    # Fill sampled results
                    for i, idx in enumerate(sampled_indices):
                        if i < sample_bag.size:  # Ensure it does not exceed bag size
                            bag_obss[bag_idx, i] = obs_list[idx]
                            bag_actions[bag_idx, i, 0] = action_list[idx]
                
        # Handle potential index overflow issues
        result_obss = []
        result_actions = []
        result_rewards = []
        result_next_obss = []
        result_next_actions = []
        result_dones = []
        
        for i in range(batch_size):
            episode_idx = episode_idxes[i][0]
            trans = transitions[i]
            
            # Ensure all indices are within valid range
            valid_trans = np.clip(trans, 0, self.max_episode_steps - 1)
            valid_next_trans = np.clip(trans + 1, 0, self.max_episode_steps)
            
            # Collect data for the current batch
            result_obss.append(self.obss[episode_idx, valid_trans])
            result_actions.append(self.actions[episode_idx, valid_trans])
            result_rewards.append(self.rewards[episode_idx, valid_trans])
            result_next_obss.append(self.obss[episode_idx, valid_next_trans])
            result_next_actions.append(self.actions[episode_idx, valid_next_trans])
            result_dones.append(self.dones[episode_idx, valid_trans])
        
        # Convert results to numpy arrays
        result_obss = np.array(result_obss)
        result_actions = np.array(result_actions)
        result_rewards = np.array(result_rewards)
        result_next_obss = np.array(result_next_obss)
        result_next_actions = np.array(result_next_actions)
        result_dones = np.array(result_dones)
        
        return (
            result_obss,
            result_actions,
            result_rewards,
            result_next_obss,
            result_next_actions,
            result_dones,
            self.episode_lengths[episode_idxes].reshape(-1),
            bag_obss,
            bag_actions,
        )
