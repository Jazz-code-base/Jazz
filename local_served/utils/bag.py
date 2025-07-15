from typing import Tuple, Union

import numpy as np


class Bag:
    """Data class for storing important observations that may have been evicted from the agent's context

    Args:
        bag_size: Size of the Bag
        obs_mask: Mask value to indicate padding in observations (will be ignored, consistently using -1)
        obs_length: Shape of observations
    """

    def __init__(self, bag_size: int, obs_mask: Union[int, float], obs_length: int):
        self.size = bag_size
        self.obs_mask = -1  # Consistently using -1 as the mask value for observations
        self.obs_length = obs_length
        # Current position in the Bag
        self.pos = 0

        self.obss, self.actions = self.make_empty_bag()

    def reset(self) -> None:
        """Reset Bag to initial state"""
        self.pos = 0
        self.obss, self.actions = self.make_empty_bag()

    def add(self, obs: np.ndarray, action: int) -> bool:
        """Add an observation-action pair to the Bag
        
        Args:
            obs: Observation value
            action: Action value
            
        Returns:
            bool: Whether addition was successful (returns False if Bag is full)
        """
        if not self.is_full:
            self.obss[self.pos] = obs
            self.actions[self.pos] = action
            self.pos += 1
            return True
        else:
            # Reject adding observation-action pair
            return False

    def export(self) -> Tuple[np.ndarray, np.ndarray]:
        """Export current valid observations and actions
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (observation array, action array)
        """
        return self.obss[: self.pos], self.actions[: self.pos]

    def make_empty_bag(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create an empty Bag
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: (observation array, action array)
        """
        # Handle image case
        if isinstance(self.obs_length, tuple):
            return (
                np.full((self.size, *self.obs_length), self.obs_mask, dtype=np.float32),
                np.full((self.size, 1), 0, dtype=np.int32)  # Use 0 instead of -1 as initial action
            )
        # Handle non-image case
        else:
            return (
                np.full((self.size, self.obs_length), self.obs_mask, dtype=np.float32),
                np.full((self.size, 1), 0, dtype=np.int32)  # Use 0 instead of -1 as initial action
            )

    @property
    def is_full(self) -> bool:
        """Check if Bag is full"""
        return self.pos >= self.size
