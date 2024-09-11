import torch
import random
import numpy as np


def set_seed(seed: int | None):
    if seed is None:
        return
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
