import unittest
import torch
from torch import nn
from rl.src.common import ConvLayer
from rl.src.dqn.dueling_dqn import DuelingDQN


class TestDuelingDQN(unittest.TestCase):
    def setUp(self):
        """Set up a DuelingDQN instance for testing."""
        input_shape = (3, 64, 64)  # Example input shape (channels, height, width)
        n_actions = 4
        hidden_layers = [128, 64]
        conv_layers = [
            ConvLayer(
                filters=32,
                kernel_size=2,
                strides=1,
                activation="relu",
                padding=0,
            ),
            ConvLayer(
                filters=32,
                kernel_size=1,
                strides=1,
                activation="relu",
                padding="same",
            ),
        ]
        self.model = DuelingDQN(input_shape, n_actions, hidden_layers, conv_layers)

    def test_initialization(self):
        """Test the initialization of DuelingDQN."""
        self.assertIsInstance(self.model.conv, nn.Sequential)
        self.assertIsInstance(self.model.fc_feature, nn.Sequential)
        self.assertIsInstance(self.model.value_stream, nn.Sequential)
        self.assertIsInstance(self.model.advantage_stream, nn.Sequential)

    def test_forward(self):
        """Test the forward pass of DuelingDQN."""
        input_tensor = torch.randn(1, *self.model.input_shape)
        output = self.model(input_tensor)
        self.assertEqual(output.size(0), 1)  # Batch size should be 1
        self.assertEqual(
            output.size(1), self.model.n_actions
        )  # Output size should match number of actions

    @unittest.skip("Skipping for now")
    def test_get_conv_output(self):
        """Test the _get_conv_output method of DuelingDQN."""
        expected_output_size = (
            32 * 64 * 64
        )  # Adjust based on the output size calculation
        output_size = self.model._get_conv_output(self.model.input_shape)
        self.assertEqual(output_size, expected_output_size)

    def test_input_shape(self):
        """Test the input_shape property of DuelingDQN."""
        self.assertEqual(self.model.input_shape, (3, 64, 64))


if __name__ == "__main__":
    unittest.main()
