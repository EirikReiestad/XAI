import random

import numpy as np

from rl.src.dqn.base import ReplayMemoryBase
from rl.src.dqn.components.types import Transition


class ReplayMemory(ReplayMemoryBase):
    """A memory buffer to store and sample transitions."""

    def push(self, *args) -> None:
        self.memory.append(Transition(*args))

    def sample(
        self, batch_size: int
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        transitions = random.sample(self.memory, batch_size)
        indices = np.arange(batch_size)
        weights = np.ones(batch_size)
        return transitions, indices, weights

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 0.01
    ) -> None:
        pass
