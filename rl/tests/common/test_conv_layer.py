import unittest
from torch import nn
from rl.src.common.conv_layer import ConvLayer


class TestConvLayer(unittest.TestCase):
    def test_invalid_padding_str(self):
        with self.assertRaises(ValueError):
            ConvLayer(16, 3, 1, "relu", "invalid")

    def test_invalid_padding_type(self):
        with self.assertRaises(ValueError):
            ConvLayer(16, 3, 1, "relu", 3.5)

    def test_invalid_activation(self):
        with self.assertRaises(ValueError):
            ConvLayer(16, 3, 1, "invalid", "same")

    def test_build_with_valid_padding(self):
        conv_layer = ConvLayer(16, 3, 1, "relu", "same")
        conv = conv_layer.build(input_channels=3)
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv.padding, (1, 1))  # For 'same' padding with stride 1

    def test_build_with_integer_padding(self):
        conv_layer = ConvLayer(16, 3, 1, "relu", 2)
        conv = conv_layer.build(input_channels=3)
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv_layer.padding, 2)

    def test_build_with_tuple_padding(self):
        conv_layer = ConvLayer(16, 3, 1, "relu", (2, 3))
        conv = conv_layer.build(input_channels=3)
        self.assertIsInstance(conv, nn.Conv2d)
        self.assertEqual(conv.padding, (2, 3))

    def test_activation(self):
        conv_layer = ConvLayer(16, 3, 1, "relu", "same")
        activation = conv_layer.build_activation()
        self.assertIsInstance(activation, nn.ReLU)

    def test_leaky_relu_activation(self):
        conv_layer = ConvLayer(16, 3, 1, "leaky_relu", "same")
        activation = conv_layer.build_activation()
        self.assertIsInstance(activation, nn.LeakyReLU)


if __name__ == "__main__":
    unittest.main()
