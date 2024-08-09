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

    def __post_init__(self):
        """Initialize the layer."""
        self.filters = int(self.filters)
        self.kernel_size = int(self.kernel_size)
        self.strides = int(self.strides)

    def build(self, input_channels: int):
        """Build the convolutional layer.
        Parameters:
            input_channels (int): The number of input channels
        """
        return nn.Conv2d(
            input_channels,
            self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=self.padding,
        )

    def build_activation(self):
        """Build the activation layer."""
        return nn.ReLU() if self.activation == "relu" else nn.LeakyReLU()
