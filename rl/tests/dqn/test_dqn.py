import unittest
import torch
import torch.nn as nn
from rl.src.dqn.dqn import DQN


class TestDQN(unittest.TestCase):
    def test_initialization(self):
        dqn = DQN(n_observations=(4,), n_actions=2, hidden_layers=[64, 32])
        self.assertIsInstance(dqn, nn.Module)
        self.assertEqual(
            len(dqn.layers), 3
        )  # input, 2 hidden, output (-1 as each module is the connection between layers)
        for layer in dqn.layers:
            self.assertIsInstance(layer, nn.Linear)

    def test_initialization_default_hidden_layers(self):
        dqn = DQN(n_observations=(4,), n_actions=2)
        self.assertEqual(len(dqn.layers), 3)  # input, 2 hidden (default), output

    def test_initialization_no_hidden_layers(self):
        dqn = DQN(n_observations=(4,), n_actions=2, hidden_layers=[])
        self.assertEqual(len(dqn.layers), 1)  # only input-to-output layer

    def test_layer_sizes(self):
        dqn = DQN(n_observations=(4,), n_actions=2, hidden_layers=[64, 32])
        self.assertEqual(dqn.layers[0].in_features, 4)
        self.assertEqual(dqn.layers[0].out_features, 64)
        self.assertEqual(dqn.layers[1].in_features, 64)
        self.assertEqual(dqn.layers[1].out_features, 32)
        self.assertEqual(dqn.layers[2].in_features, 32)
        self.assertEqual(dqn.layers[2].out_features, 2)

    def test_forward_pass(self):
        dqn = DQN(n_observations=(4,), n_actions=2, hidden_layers=[64, 32])
        input_tensor = torch.randn(1, 4)
        output = dqn(input_tensor)
        self.assertEqual(output.shape, (1, 2))

    def test_forward_pass_batch(self):
        dqn = DQN(n_observations=(4,), n_actions=2, hidden_layers=[64, 32])
        input_tensor = torch.randn(32, 4)  # Batch size of 32
        output = dqn(input_tensor)
        self.assertEqual(output.shape, (32, 2))

    def test_forward_pass_2d_input(self):
        dqn = DQN(n_observations=(3, 3), n_actions=2, hidden_layers=[64, 32])
        input_tensor = torch.randn(1, 3, 3)
        output = dqn(input_tensor)
        self.assertEqual(output.shape, (1, 2))

    def test_build_network(self):
        dqn = DQN(n_observations=(4,), n_actions=2)
        layers = dqn._build_network(10, [20, 30], 5)
        self.assertEqual(len(layers), 3)
        self.assertEqual(layers[0].in_features, 10)
        self.assertEqual(layers[0].out_features, 20)
        self.assertEqual(layers[1].in_features, 20)
        self.assertEqual(layers[1].out_features, 30)
        self.assertEqual(layers[2].in_features, 30)
        self.assertEqual(layers[2].out_features, 5)

    def test_build_network_no_hidden_layers(self):
        dqn = DQN(n_observations=(4,), n_actions=2)
        layers = dqn._build_network(10, [], 5)
        self.assertEqual(len(layers), 1)
        self.assertEqual(layers[0].in_features, 10)
        self.assertEqual(layers[0].out_features, 5)


if __name__ == "__main__":
    unittest.main()
