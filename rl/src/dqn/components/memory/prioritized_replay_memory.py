import numpy as np

from rl.src.dqn.base import ReplayMemoryBase
from rl.src.dqn.components.types import Transition


class PrioritizedReplayMemory(ReplayMemoryBase):
    def __init__(self, capacity: int, alpha: float = 0.6) -> None:
        super().__init__(capacity)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha

    def push(self, *args) -> None:
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
        else:
            self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = max_priority**self.alpha
        self.position = (self.position + 1) % self.capacity

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[list[Transition], np.ndarray, np.ndarray]:
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[: self.position]

        if priorities.sum() == 0:
            probabilities = np.ones_like(priorities) / len(priorities)
        else:
            probabilities = priorities / priorities.sum()

        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        transitions = [self.memory[idx] for idx in indices]
        total = len(self.memory)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return transitions, indices, weights

    def update_priorities(
        self, indices: np.ndarray, td_errors: np.ndarray, epsilon: float = 0.01
    ) -> None:
        priorities = (np.abs(td_errors) + epsilon) ** self.alpha
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
