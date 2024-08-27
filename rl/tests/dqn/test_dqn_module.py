import unittest
from unittest.mock import MagicMock, patch

import torch
from torch import nn
from torch.optim.adamw import AdamW

from rl.src.dqn.dqn_module import DQNModule

from rl.src.dqn.replay_memory import ReplayMemory


class TestDQNModule(unittest.TestCase):
    def setUp(self):
        """Set up for DQNModule tests."""
        self.observation_shape = (3, 84, 84)  # Example shape
        self.n_actions = 4  # Example number of actions
        self.dqn_module = DQNModule(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
            hidden_layers=[128, 128],
            conv_layers=None,
            seed=42,
        )

    @unittest.skip("Look at later")
    @patch("rl.src.dqn.dqn_module.DQN")  # Adjust based on your module's import path
    @patch("rl.src.dqn.dqn_module.DuelingDQN")
    def test_initialization(self, mock_dueling_dqn, mock_dqn):
        """Test the initialization of DQNModule."""
        mock_dqn.return_value = MagicMock(spec=nn.Module)
        mock_dueling_dqn.return_value = MagicMock(spec=nn.Module)
        dqn_module = DQNModule(
            observation_shape=self.observation_shape,
            n_actions=self.n_actions,
            hidden_layers=[128, 128],
            conv_layers=None,
        )
        self.assertIsInstance(dqn_module.policy_net, nn.Module)
        self.assertIsInstance(dqn_module.target_net, nn.Module)
        self.assertIsInstance(dqn_module.optimizer, AdamW)
        self.assertIsInstance(dqn_module.memory, ReplayMemory)

    def test_select_action(self):
        """Test action selection."""
        state = torch.zeros(self.observation_shape, device="cpu")
        action = self.dqn_module.select_action(state)
        self.assertEqual(action.shape, (1, 1))
        self.assertEqual(action.dtype, torch.long)

    def test_train(self):
        """Test the training process."""
        state = torch.zeros(self.observation_shape, device="cpu")
        action = torch.tensor([[1]], device="cpu")
        observation = torch.zeros(*self.observation_shape, device="cpu")
        reward = 1.0
        terminated = False
        truncated = False
        done, next_state = self.dqn_module.train(
            state, action, observation, reward, terminated, truncated
        )
        self.assertFalse(done)
        self.assertIsInstance(next_state, torch.Tensor)
        if next_state is None:
            raise AssertionError("next_state is None")
        self.assertEqual(next_state.shape, self.observation_shape)

    def test_invalid_state_shape(self):
        """Test invalid state shape."""
        state = torch.zeros((1, 3, 84, 84), device="cpu")
        with self.assertRaises(ValueError):
            self.dqn_module.select_action(state)

    def test_optimize_model(self):
        """Test optimization step."""
        state = torch.zeros(self.observation_shape, device="cpu")
        action = torch.randint(0, self.n_actions, (5, 1), device="cpu")
        next_state = torch.zeros(self.observation_shape, device="cpu")
        reward = 5
        done = False

        for _ in range(10):  # Fill replay memory with some transitions
            self.dqn_module.train(state, action, next_state, reward, done, done)
        # Optimize model
        self.dqn_module._optimize_model()  # Test if no exceptions are raised


if __name__ == "__main__":
    unittest.main()
