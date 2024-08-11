import unittest
import torch
from rl.src.dqn.replay_memory import ReplayMemory
from rl.src.dqn.utils import Transition


class TestReplayMemory(unittest.TestCase):
    def setUp(self):
        """Set up a ReplayMemory instance for testing."""
        self.capacity = 5
        self.memory = ReplayMemory(self.capacity)

    def test_initialization(self):
        """Test initialization of ReplayMemory."""
        self.assertEqual(len(self.memory), 0)
        self.assertEqual(self.memory.capacity, self.capacity)

    def test_push(self):
        """Test pushing transitions into ReplayMemory."""
        state = torch.tensor([1.0])
        action = torch.tensor([0])
        next_state = torch.tensor([2.0])
        reward = torch.tensor([1.0])

        self.memory.push(state, action, next_state, reward)
        self.assertEqual(len(self.memory), 1)
        self.assertEqual(
            self.memory.memory[0], Transition(state, action, next_state, reward)
        )

    def test_push_over_capacity(self):
        """Test pushing more transitions than the capacity."""
        for i in range(self.capacity + 2):
            state = torch.tensor([float(i)])
            action = torch.tensor([i % 2])
            next_state = torch.tensor([float(i + 1)])
            reward = torch.tensor([float(i % 2)])
            self.memory.push(state, action, next_state, reward)

        # Capacity should be enforced, so only the latest `capacity` transitions should be kept
        self.assertEqual(len(self.memory), self.capacity)
        self.assertEqual(
            self.memory.memory[0],
            Transition(
                torch.tensor([2.0]),
                torch.tensor([0]),
                torch.tensor([3.0]),
                torch.tensor([0]),
            ),
        )

    def test_sample(self):
        """Test sampling transitions from ReplayMemory."""
        for i in range(self.capacity):
            state = torch.tensor([float(i)])
            action = torch.tensor([i % 2])
            next_state = torch.tensor([float(i + 1)])
            reward = torch.tensor([float(i % 2)])
            self.memory.push(state, action, next_state, reward)

        sampled_transitions = self.memory.sample(batch_size=3)
        self.assertEqual(len(sampled_transitions), 3)
        self.assertTrue(all(isinstance(t, Transition) for t in sampled_transitions))

    def test_sample_more_than_available(self):
        """Test sampling more transitions than available in memory."""
        for i in range(self.capacity):
            state = torch.tensor([float(i)])
            action = torch.tensor([i % 2])
            next_state = torch.tensor([float(i + 1)])
            reward = torch.tensor([float(i % 2)])
            self.memory.push(state, action, next_state, reward)

        with self.assertRaises(ValueError):
            self.memory.sample(batch_size=self.capacity + 1)

    def test_len(self):
        """Test the length of ReplayMemory."""
        self.assertEqual(len(self.memory), 0)
        self.memory.push(
            torch.tensor([1.0]),
            torch.tensor([0]),
            torch.tensor([2.0]),
            torch.tensor([1.0]),
        )
        self.assertEqual(len(self.memory), 1)


if __name__ == "__main__":
    unittest.main()
