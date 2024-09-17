import unittest
import random
import numpy as np
from rl.src.dqn.components.types import Transition
from rl.src.dqn.components.memory import (
    ReplayMemory,
)


class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        self.memory = ReplayMemory(capacity=100)
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
        self.assertTrue(np.all(weights == 1.0))

    def test_update_priorities(self):
        # Test that the method runs without error (it's a no-op)
        indices = np.array([0])
        td_errors = np.array([0.5])
        self.memory.update_priorities(indices, td_errors)
        # No assert needed as method is a no-op


if __name__ == "__main__":
    unittest.main()
