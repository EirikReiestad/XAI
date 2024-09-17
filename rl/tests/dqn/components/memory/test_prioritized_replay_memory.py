import unittest
import numpy as np
from rl.src.dqn.base import ReplayMemoryBase
from rl.src.dqn.components.types import Transition
from rl.src.dqn.components.memory import (
    PrioritizedReplayMemory,
)


class TestPrioritizedReplayMemory(unittest.TestCase):
    def setUp(self):
        self.memory = PrioritizedReplayMemory(capacity=100, alpha=0.6)
        self.transition = Transition(
            state=np.array([0.0]),
            action=1,
            next_state=np.array([1.0]),
            reward=1.0,
        )

    def test_push(self):
        self.memory.push(
            self.transition.state,
            self.transition.action,
            self.transition.next_state,
            self.transition.reward,
        )
        self.assertEqual(len(self.memory.memory), 1)
        self.assertEqual(self.memory.priorities[0], 1.0**self.memory.alpha)

    def test_sample(self):
        for _ in range(10):
            self.memory.push(
                self.transition.state,
                self.transition.action,
                self.transition.next_state,
                self.transition.reward,
            )

        transitions, indices, weights = self.memory.sample(batch_size=5)
        self.assertEqual(len(transitions), 5)
        self.assertEqual(len(indices), 5)
        self.assertEqual(weights.shape[0], 5)

    def test_update_priorities(self):
        self.memory.push(
            self.transition.state,
            self.transition.action,
            self.transition.next_state,
            self.transition.reward,
        )
        indices = np.array([0])
        td_errors = np.array([0.5])
        self.memory.update_priorities(indices, td_errors)
        expected_priority = (np.abs(0.5) + 0.01) ** self.memory.alpha
        expected_priority = np.float32(expected_priority)
        self.assertEqual(
            round(self.memory.priorities[0], 4), round(expected_priority, 4)
        )


if __name__ == "__main__":
    unittest.main()
