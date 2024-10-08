from abc import ABC, abstractmethod
from collections import deque

import numpy as np

from rl.src.dqn.components.types import Transition


class ReplayMemoryBase(ABC):
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)

    @abstractmethod
    def push(self, *args) -> None:
        raise NotImplementedError

    @abstractmethod
    def sample(
        self, batch_size: int
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 0.01
    ) -> None:
        pass

    def __len__(self) -> int:
        return len(self.memory)
