import unittest
import torch
from unittest.mock import patch

from your_dqn_module_file import DQNModule, Transition  # Replace with your file path


class TestDQNModule(unittest.TestCase):

    def setUp(self):
        self.n_observations = 10
        self.n_actions = 5
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.dqn_module = DQNModule(self.n_observations, self.n_actions)

    def test_select_action_epsilon_greedy(self):
        # Test with high epsilon for random exploration
        self.dqn_module.eps_start = 0.9
        state = torch.randn(1, self.n_observations, device=self.device)
        action_counts = torch.zeros(self.n_actions)
        for _ in range(1000):
            action_counts[self.dqn_module.select_action(state).item()] += 1
        # Allow for some randomness
        self.assertAlmostEqual(action_counts.mean().item(), 0.2, delta=0.1)

        # Test with low epsilon for greedy selection
        self.dqn_module.eps_start = 0.1
        self.dqn_module.eps_end = 0.1
        action_counts = torch.zeros(self.n_actions)
        for _ in range(1000):
            action_counts[self.dqn_module.select_action(state).item()] += 1
        # Expect high probability for best action
        self.assertGreater(action_counts.max().item(), 0.8)

    def test_select_action_no_exploration(self):
        self.dqn_module.eps_start = 0.0
        self.dqn_module.eps_end = 0.0
        state = torch.randn(1, self.n_observations, device=self.device)
        # Simulate setting policy net output (replace with your actual logic)
        self.dqn_module.policy_net = torch.nn.Sequential(
            torch.nn.Linear(self.n_observations, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, self.n_actions),
        )
        self.dqn_module.policy_net.to(self.device)
        q_values = self.dqn_module.policy_net(state)
        expected_action = torch.argmax(q_values, dim=1)
        action = self.dqn_module.select_action(state)
        self.assertEqual(action.item(), expected_action.item())

    @patch.object(DQNModule, 'memory')
    def test_optimize_model_empty_memory(self, mock_memory):
        mock_memory.sample.return_value = []
        self.dqn_module.optimize_model()
        # No gradient update if memory is empty
        self.assertFalse(hasattr(self.dqn_module.optimizer, 'zero_grad'))

    @patch.object(DQNModule, 'memory')
    def test_optimize_model(self, mock_memory):
        # Create a mock batch of transitions
        batch_size = 32
        state_batch = torch.rand(
            batch_size, self.n_observations, device=self.device)
        action_batch = torch.randint(
            0, self.n_actions, (batch_size,), device=self.device)
        reward_batch = torch.rand(batch_size, device=self.device)
        next_state_batch = torch.rand(
            batch_size, self.n_observations, device=self.device)
        non_final_mask = torch.ones(
            batch_size, dtype=torch.bool, device=self.device)
        mock_memory.sample.return_value = Transition(
            *zip(state_batch, action_batch, next_state_batch, reward_batch))

        # Mock target network output for expected state values
        with torch.no_grad():
            mock_target_values = torch.rand(batch_size, device=self.device)
