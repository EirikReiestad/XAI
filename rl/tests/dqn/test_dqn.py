import unittest
import torch
import os
from src.dqn.dqn import DQN


class TestDQN(unittest.TestCase):

    def setUp(self):
        self.input_dim = 4
        self.output_dim = 2
        self.hidden_dims = [64, 64]
        self.dqn = DQN(self.input_dim, self.output_dim, self.hidden_dims)

    def test_initialization(self):
        self.assertIsInstance(self.dqn, DQN)
        self.assertEqual(self.dqn.output_dim, self.output_dim)
        self.assertEqual(len(self.dqn.replay_buffer), 0)

    def test_choose_action(self):
        state = torch.randn(1, self.input_dim)
        action = self.dqn.choose_action(state)
        self.assertTrue(0 <= action < self.output_dim)

    def test_update_epsilon(self):
        initial_epsilon = self.dqn.epsilon
        self.dqn._update_epsilon()
        self.assertLess(self.dqn.epsilon, initial_epsilon)

    def test_update(self):
        state = torch.randn(1, self.input_dim)
        next_state = torch.randn(1, self.input_dim)
        self.dqn.update(state, 0, 1.0, next_state, False)
        self.assertEqual(len(self.dqn.replay_buffer), 1)

    def test_save_and_load_model(self):
        path = "test_model.pth"
        self.dqn.save(path)
        self.assertTrue(os.path.exists(path))
        new_dqn = DQN(self.input_dim, self.output_dim, self.hidden_dims)
        new_dqn.load(path)
        self.assertTrue(os.path.exists(path))
        os.remove(path)

    def test_train(self):
        states = torch.randn(32, self.input_dim)
        actions = torch.randint(0, self.output_dim, (32,))
        rewards = torch.rand(32)
        next_states = torch.randn(32, self.input_dim)
        dones = torch.randint(0, 2, (32,), dtype=torch.bool)
        self.dqn.train(states, actions, rewards, next_states, dones)
        self.assertEqual(self.dqn.train_step, 1)


if __name__ == '__main__':
    unittest.main()
