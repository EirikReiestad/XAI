from abc import ABC, abstractmethod
import numpy as np
from collections import deque

from rl.src.dqn.utils import Transition


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

    def __len__(self) -> int:
        return len(self.memory)
