import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import torch
import gymnasium as gym
from rl.src.dqn import DQN
from rl.src.dqn.components.types import RolloutReturn


class TestDQN(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(spec=gym.Env)
        self.policy = MagicMock()
        self.policy_net = MagicMock()
        self.target_net = MagicMock()
        self.env.action_space.n = 4
        self.env.observation_space.shape = (4,)
        self.dqn = DQN(env=self.env, policy=self.policy, dueling=False, double=False)
        self.dqn.policy_net = self.policy_net
        self.dqn.target_net = self.target_net
        self.dqn.optimizer = MagicMock()
        self.dqn.memory = MagicMock()
        self.dqn.memory.sample = MagicMock(
            return_value=(
                [
                    (
                        torch.zeros(1, 4),
                        torch.tensor(0),
                        torch.zeros(1, 4),
                        torch.tensor(1.0),
                    )
                ],
                [0],
                [1.0],
            )
        )

    def test_learn(self):
        self.dqn._collect_rollout = MagicMock(return_value=MagicMock())
        self.dqn.learn(total_timesteps=1)
        self.dqn._collect_rollout.assert_called_once()

    def test_collect_rollout(self):
        self.env.reset.return_value = (np.zeros((4,)), {})
        self.env.step.return_value = (np.zeros((4,)), 1.0, False, False, {})
        rollout_return = self.dqn._collect_rollout()
        self.assertIsInstance(rollout_return, RolloutReturn)

    def test_predict(self):
        self.policy_net.return_value = torch.tensor([[1.0]])
        state = torch.zeros(1, 4)
        action = self.dqn.predict(state)
        self.assertEqual(action.item(), 0)

    def test_get_q_values(self):
        self.policy_net.return_value = torch.tensor([0.0, 1.0, 2.0, 3.0])
        states = np.zeros((2, 2, 4))
        q_values = self.dqn.get_q_values(states)
        self.assertEqual(q_values.shape, (2, 2, 4))


if __name__ == "__main__":
    unittest.main()
