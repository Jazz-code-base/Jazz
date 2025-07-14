import torch
import numpy as np
import random
import os
from typing import Optional, Tuple


class RNG:
    rng: np.random.Generator = None


def set_global_seed(seed: int) -> None:
    """Sets seed for PyTorch, NumPy, and random.

    Args:
        seed: The random seed to use.
    """
    random.seed(seed)
    tseed = random.randint(1, 1e6)
    npseed = random.randint(1, 1e6)
    ospyseed = random.randint(1, 1e6)
    torch.manual_seed(tseed)
    np.random.seed(npseed)
    os.environ["PYTHONHASHSEED"] = str(ospyseed)
    RNG.rng = np.random.Generator(np.random.PCG64(seed=seed))
