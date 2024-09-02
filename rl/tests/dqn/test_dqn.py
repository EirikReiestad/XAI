import unittest
import torch
from unittest.mock import patch
from rl.src.dqn.dqn_module import DQNModule, device


class TestDQNModule(unittest.TestCase):
    def setUp(self):
        self.rgb_shape = (1, 3, 84, 84)  # RGB image
        self.grayscale_shape = (1, 84, 84)  # Grayscale image
        self.flat_shape = (1, 100)  # 1D array
        self.n_actions = 4

    def test_initialization_rgb(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        self.assertEqual(dqn.observation_shape, self.rgb_shape)
        self.assertEqual(dqn.n_actions, self.n_actions)

    def test_initialization_grayscale(self):
        dqn = DQNModule(self.grayscale_shape, self.n_actions)
        self.assertEqual(dqn.observation_shape, self.grayscale_shape)
        self.assertEqual(dqn.n_actions, self.n_actions)

    def test_initialization_flat(self):
        dqn = DQNModule(self.flat_shape, self.n_actions)
        self.assertEqual(dqn.observation_shape, self.flat_shape)
        self.assertEqual(dqn.n_actions, self.n_actions)

    def test_select_action_rgb(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        state = torch.randn(*self.rgb_shape).to(device)
        action = dqn.select_action(state)
        self.assertEqual(action.shape, (1, 1))
        self.assertTrue(0 <= action.item() < self.n_actions)

    def test_select_action_grayscale(self):
        dqn = DQNModule(self.grayscale_shape, self.n_actions)
        state = torch.randn(*self.grayscale_shape).to(device)
        action = dqn.select_action(state)
        self.assertEqual(action.shape, (1, 1))
        self.assertTrue(0 <= action.item() < self.n_actions)

    def test_select_action_flat(self):
        dqn = DQNModule(self.flat_shape, self.n_actions)
        state = torch.randn(*self.flat_shape).to(device)
        action = dqn.select_action(state)
        self.assertEqual(action.shape, (1, 1))
        self.assertTrue(0 <= action.item() < self.n_actions)

    def test_select_action_invalid_shape(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        invalid_state = torch.randn(1, 4, 84, 84).to(device)
        with self.assertRaises(ValueError):
            dqn.select_action(invalid_state)

    def test_train_rgb(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        state = [torch.randn(*self.rgb_shape).to(device)]
        action = [torch.tensor([[0]], device=device)]
        observation = [torch.randn(*self.rgb_shape).to(device)]
        reward = [torch.tensor(1.0, device=device, dtype=torch.float32)]
        terminated = [False]
        truncated = [False]

        dqn.train(state, action, observation, reward, terminated, truncated)

        self.assertTrue(torch.equal(state[0], observation[0]))
        self.assertEqual(len(dqn.memory), 1)

    def test_train_grayscale(self):
        dqn = DQNModule(self.grayscale_shape, self.n_actions)
        state = [torch.randn(*self.grayscale_shape).to(device)]
        action = [torch.tensor([[0]], device=device)]
        observation = [torch.randn(*self.grayscale_shape).to(device)]
        reward = [torch.tensor(1.0, device=device, dtype=torch.float32)]
        terminated = [False]
        truncated = [False]

        dqn.train(state, action, observation, reward, terminated, truncated)

        self.assertTrue(torch.equal(state[0], observation[0]))
        self.assertEqual(len(dqn.memory), 1)

    def test_train_flat(self):
        dqn = DQNModule(self.flat_shape, self.n_actions)
        state = [torch.randn(*self.flat_shape).to(device)]
        action = [torch.tensor([[0]], device=device)]
        observation = [torch.randn(*self.flat_shape).to(device)]
        reward = [torch.tensor(1.0, device=device, dtype=torch.float32)]
        terminated = [False]
        truncated = [False]

        dqn.train(state, action, observation, reward, terminated, truncated)

        self.assertTrue(torch.equal(state[0], observation[0]))
        self.assertEqual(len(dqn.memory), 1)

    def test_train_invalid_shapes(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        invalid_state = [torch.randn(1, 4, 84, 84).to(device)]
        action = [torch.tensor([[0]], device=device)]
        observation = [torch.randn(1, *self.rgb_shape).to(device)]
        reward = [torch.tensor(1.0, device=device, dtype=torch.float32)]
        terminated = [False]
        truncated = [False]

        with self.assertRaises(ValueError):
            dqn.train(invalid_state, action, observation, reward, terminated, truncated)

        invalid_observation = [torch.randn(1, 4, 84, 84).to(device)]
        with self.assertRaises(ValueError):
            dqn.train(
                observation, action, invalid_observation, reward, terminated, truncated
            )

    @patch.object(DQNModule, "_optimize_model")
    @patch.object(DQNModule, "_soft_update_target_net")
    def test_train_calls_optimize_and_update(self, mock_update, mock_optimize):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        state = [torch.randn(*self.rgb_shape).to(device)]
        action = [torch.tensor([[0]], device=device)]
        observation = [torch.randn(*self.rgb_shape).to(device)]
        reward = [torch.tensor(1.0, device=device, dtype=torch.float32)]
        terminated = [False]
        truncated = [False]

        dqn.train(state, action, observation, reward, terminated, truncated)

        mock_optimize.assert_called_once()
        mock_update.assert_called_once()

    @unittest.skip("Wrong test?")
    def test_soft_update_target_net(self):
        dqn = DQNModule(self.rgb_shape, self.n_actions)
        initial_target_state = dqn.target_net.state_dict()
        dqn._soft_update_target_net()
        updated_target_state = dqn.target_net.state_dict()

        for key in initial_target_state:
            self.assertFalse(
                torch.equal(initial_target_state[key], updated_target_state[key])
            )


if __name__ == "__main__":
    unittest.main()
