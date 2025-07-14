import numpy as np
from typing import List, Optional, Union


class RunningAverage:
    """A class to keep track of a running average of a value.

    Args:
        window_size: The number of values to keep track of. If None, keep track of all values.
    """

    def __init__(self, window_size: Optional[int] = None):
        self.window_size = window_size
        self.values: List[float] = []
        self.sum = 0.0

    def add(self, value: Union[float, np.ndarray]) -> None:
        """Add a value to the running average.

        Args:
            value: The value to add.
        """
        if isinstance(value, np.ndarray):
            value = float(value)
        self.values.append(value)
        self.sum += value
        if self.window_size is not None and len(self.values) > self.window_size:
            self.sum -= self.values.pop(0)

    def get(self) -> float:
        """Get the current running average.

        Returns:
            The current running average.
        """
        if not self.values:
            return 0.0
        return self.sum / len(self.values)

    def reset(self) -> None:
        """Reset the running average."""
        self.values = []
        self.sum = 0.0 