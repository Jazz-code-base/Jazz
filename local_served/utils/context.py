from typing import Tuple, Union
import numpy as np
import torch

class RNG:
    """Random Number Generator class"""
    rng = np.random.default_rng()

# noinspection PyAttributeOutsideInit
class Context:
    """Context class for storing agent history (stores up to max_length transitions)

    Args:
        context_length: Maximum number of transitions to store
        obs_mask: Mask value for observations (now using -1)
        num_actions: Number of possible actions in the environment
        env_obs_length: Dimension of the observation (assuming 1D array)
        init_hidden: Initial value for hidden state (used for RNN)
    """

    def __init__(
        self,
        context_length: int,
        obs_mask: int,  # This parameter will be ignored, consistently using -1
        num_actions: int,
        env_obs_length: int,
        init_hidden: Tuple[torch.Tensor] = None,
    ):
        self.max_length = context_length
        self.env_obs_length = env_obs_length
        self.num_actions = num_actions
        self.obs_mask = -1  # Consistently using -1 as the mask value for observations
        self.reward_mask = 0.0
        self.done_mask = True
        self.timestep = 0
        self.init_hidden = init_hidden

        # Ensure RNG is initialized
        if not hasattr(RNG, 'rng') or RNG.rng is None:
            RNG.rng = np.random.default_rng()

    def reset(self, obs: np.ndarray):
        """Reset context to initial state"""
        # Handle image case
        if isinstance(self.env_obs_length, tuple):
            self.obs = np.full(
                [self.max_length, *self.env_obs_length],
                self.obs_mask,
                dtype=np.float32,  # Changed to float32 to support -1 values
            )
        else:
            self.obs = np.full([self.max_length, self.env_obs_length], self.obs_mask, dtype=np.float32)
        
        # Set initial observation
        self.obs[0] = obs

        # Fill actions with random integers, consistent with original DTQN-main
        self.action = RNG.rng.integers(self.num_actions, size=(self.max_length, 1), dtype=np.int32)
        self.reward = np.full_like(self.action, self.reward_mask, dtype=np.float32)
        self.done = np.full_like(self.reward, self.done_mask, dtype=np.int32)
        self.hidden = self.init_hidden
        self.timestep = 0

    def add_transition(
        self, o: np.ndarray, a: int, r: float, done: bool
    ) -> Tuple[Union[np.ndarray, None], Union[int, None]]:
        """Add a complete transition. If the context is full, evict the oldest transition"""
        self.timestep += 1
        self.obs = self.roll(self.obs)
        self.action = self.roll(self.action)
        self.reward = self.roll(self.reward)
        self.done = self.roll(self.done)

        t = min(self.timestep, self.max_length - 1)

        # If we need to evict an observation, return it so it can potentially be added to the bag
        evicted_obs = None
        evicted_action = None
        if self.is_full:
            evicted_obs = self.obs[t].copy()
            evicted_action = self.action[t].item()  # Convert to Python scalar

        self.obs[t] = o
        self.action[t] = np.array([a], dtype=np.int32)
        self.reward[t] = np.array([r], dtype=np.float32)
        self.done[t] = np.array([done], dtype=np.int32)

        return evicted_obs, evicted_action

    def get_valid_mask(self) -> np.ndarray:
        """Get mask for valid data (non-padded portion)
        
        Returns:
            np.ndarray: Boolean mask array, True indicates valid data
        """
        valid_length = min(self.timestep + 1, self.max_length)
        mask = np.zeros(self.max_length, dtype=bool)
        mask[:valid_length] = True
        return mask

    def get_valid_length(self) -> int:
        """Get the valid length (length of non-padded portion)
        
        Returns:
            int: Length of valid data
        """
        return min(self.timestep + 1, self.max_length)

    def roll(self, arr: np.ndarray) -> np.ndarray:
        """Helper function for inserting at the end of array. If context is full,
        we replace the first element with the new element and 'roll' the new element to the end of array
        """
        return np.roll(arr, -1, axis=0) if self.timestep >= self.max_length else arr

    @property
    def is_full(self) -> bool:
        """Check if the context is full"""
        return self.timestep >= self.max_length

    @staticmethod
    def context_like(context):
        """Create a new context that mimics the provided context"""
        return Context(
            context.max_length,
            context.obs_mask,  # This value will be ignored, consistently using -1
            context.num_actions,
            context.env_obs_length,
            init_hidden=context.init_hidden,
        )
