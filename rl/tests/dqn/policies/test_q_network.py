import unittest
import torch
import numpy as np
from gymnasium import spaces
from rl.src.dqn.policies import (
    QNetwork,
)


class TestQNetwork(unittest.TestCase):
    def setUp(self):
        observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(2)
        self.network = QNetwork(
            observation_space, action_space, hidden_layers=[16, 16], dueling=True
        )

    def test_forward(self):
        x = torch.rand(1, 4)
        output = self.network(x)
        self.assertEqual(output.shape, (1, 2))

    def test_observation_size(self):
        observation_space = spaces.Box(low=0, high=1, shape=(4, 4), dtype=np.float32)
        network = QNetwork(
            observation_space, spaces.Discrete(2), hidden_layers=[16, 16]
        )
        self.assertEqual(network._observation_size(observation_space), 16)

    def test_invalid_observation_size(self):
        observation_space = spaces.Box(low=0, high=1, shape=(0,), dtype=np.float32)
        with self.assertRaises(ValueError):
            QNetwork(observation_space, spaces.Discrete(2), hidden_layers=[16, 16])

    def test_dueling_network(self):
        observation_space = spaces.Box(low=0, high=1, shape=(4,), dtype=np.float32)
        action_space = spaces.Discrete(3)
        network = QNetwork(
            observation_space, action_space, hidden_layers=[16, 16], dueling=True
        )
        x = torch.rand(1, 4)
        output = network(x)
        self.assertEqual(output.shape, (1, 3))


if __name__ == "__main__":
    unittest.main()
