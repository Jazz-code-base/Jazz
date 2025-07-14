class LinearAnneal:
    """A class to linearly anneal a value from start to end over a number of steps.

    Args:
        start: The starting value.
        end: The ending value.
        steps: The number of steps to anneal over.
    """

    def __init__(self, start: float, end: float, steps: int):
        self.start = start
        self.end = end
        self.steps = steps
        self.step = 0

    def __call__(self) -> float:
        """Get the current value.

        Returns:
            The current value.
        """
        if self.step >= self.steps:
            return self.end
        value = self.start + (self.end - self.start) * (self.step / self.steps)
        self.step += 1
        return value

    def reset(self) -> None:
        """Reset the annealing."""
        self.step = 0 