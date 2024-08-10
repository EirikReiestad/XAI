from dataclasses import dataclass

from torch import nn


@dataclass
class ConvLayer:
    """Convolutional layer configuration."""

    filters: int
    kernel_size: int
    strides: int
    activation: str
    padding: str

    def build(self, input_channels: int):
        """Construct the convolutional layer followed by the activation.

        Args:
            input_channels (int): Number of input channels.

        Returns:
            nn.Sequential: A sequential model containing the convolutional layer and activation.
        """
        return nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding,
        )

    def build_activation(self):
        """Retrieve the activation function based on the specified type.

        Returns:
            nn.Module: The appropriate activation layer.
        """
        if self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "leaky_relu":
            return nn.LeakyReLU()
        else:
            raise ValueError(f"Unsupported activation type: {self.activation}")
