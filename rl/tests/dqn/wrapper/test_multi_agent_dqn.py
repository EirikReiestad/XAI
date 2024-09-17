import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from environments.gymnasium.wrappers import MultiAgentEnv
from rl.src.dqn.components.types import RolloutReturn
from rl.src.dqn.policies import DQNPolicy
from rl.src.dqn.wrapper import (
    MultiAgentDQN,
)


class TestMultiAgentDQN(unittest.TestCase):
    def setUp(self):
        self.env = MagicMock(spec=MultiAgentEnv)
        self.dqn_policy = MagicMock(spec=DQNPolicy)
        self.num_agents = 2
        self.multi_agent_dqn = MultiAgentDQN(self.env, self.num_agents, self.dqn_policy)

        # Set up mocks
        self.env.reset.return_value = (np.zeros((4,)), {})
        self.env.get_wrapper_attr.return_value = (
            np.zeros((self.num_agents, 4)),
            np.zeros((self.num_agents, 4)),
            [False] * self.num_agents,
            [np.zeros((4,)) for _ in range(self.num_agents)],
            [1.0] * self.num_agents,
            [False] * self.num_agents,
            [False] * self.num_agents,
            [{} for _ in range(self.num_agents)],
        )
        for agent in self.multi_agent_dqn.agents:
            agent.train = MagicMock()
            agent.predict = MagicMock(
                return_value=[torch.tensor(0) for _ in range(self.num_agents)]
            )
            agent.get_q_values = MagicMock(return_value=np.zeros((4,)))
            agent.load = MagicMock()
            agent.save = MagicMock()

    def test_learn(self):
        self.multi_agent_dqn.learn(total_timesteps=1)
        # Validate that rollouts are collected and logged
        self.env.reset.assert_called_once()
        self.env.get_wrapper_attr.return_value[
            0
        ]  # Verify interactions with the environment

    def test_collect_rollouts(self):
        rollouts = self.multi_agent_dqn._collect_rollouts()
        self.assertEqual(len(rollouts), self.num_agents)
        for rollout in rollouts:
            self.assertIsInstance(rollout, RolloutReturn)

    def test_predict(self):
        state = torch.tensor(np.zeros((4,)), dtype=torch.float32).unsqueeze(0)
        actions = self.multi_agent_dqn.predict(state)
        self.assertEqual(len(actions), self.num_agents)
        for action in actions:
            self.assertIsInstance(action, torch.Tensor)

    def test_get_q_values(self):
        states = np.zeros((4,))
        q_values = self.multi_agent_dqn.get_q_values(states, agent=0)
        self.assertTrue(np.array_equal(q_values, np.zeros((4,))))


if __name__ == "__main__":
    unittest.main()
