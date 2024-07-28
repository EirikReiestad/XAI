import unittest
import torch
import numpy as np
from rl.src.dqn.dqn_module import DQNModule


class TestDQNModule(unittest.TestCase):

    def setUp(self):
        self.observation_shape = torch.Size([1, 3, 64, 64])
        self.n_actions = 2
        self.module = DQNModule(self.observation_shape,
                                self.n_actions, seed=42)

    def test_initialization(self):
        self.assertEqual(self.module.n_actions, self.n_actions)
        self.assertEqual(self.module.observation_shape, self.observation_shape)
        self.assertEqual(self.module.steps_done, 0)

    def test_select_action_exploration(self):
        state = torch.full(self.observation_shape, 0.0)
        action = self.module.select_action(state)
        self.assertEqual(action.size(), (1, 1))
        self.assertTrue(0 <= action.item() < self.n_actions)

    def test_optimize_model_no_memory(self):
        # Ensure optimize_model does not crash when memory is insufficient
        self.module.optimize_model()
        self.assertTrue(True)  # If no exception, the test passes

    def test_memory_push(self):
        state = torch.full(self.observation_shape, 0.0)
        action = torch.tensor([[0]], dtype=torch.long)
        next_state = torch.full(self.observation_shape, 0.0)
        reward = torch.tensor([1.0])

        self.module.memory.push(state, action, next_state, reward)
        self.assertEqual(len(self.module.memory), 1)

    def test_train(self):
        state = torch.full(self.observation_shape, 0.0)
        action = torch.tensor([[0]], dtype=torch.long)
        observation = torch.full(self.observation_shape, 0.0)
        reward = 1.0
        terminated = False
        truncated = False

        done, next_state = self.module.train(
            state, action, observation, reward, terminated, truncated)
        self.assertFalse(done)
        self.assertEqual(next_state.size(), self.observation_shape)

    def test_train_terminated(self):
        state = torch.full(self.observation_shape, 0.0)
        action = torch.tensor([[0]], dtype=torch.long)
        observation = torch.full(self.observation_shape, 0.0)
        reward = 1.0
        terminated = True
        truncated = False

        done, next_state = self.module.train(
            state, action, observation, reward, terminated, truncated)
        self.assertTrue(done)
        self.assertIsNone(next_state)


if __name__ == '__main__':
    unittest.main()
