from dataclasses import dataclass

from torch import nn


@dataclass
class ConvLayer:
    """Convolutional layer configuration."""

    filters: int
    kernel_size: int
    strides: int
    activation: str
    padding: str | int | tuple[int, int]

    def __post_init__(self):
        if self.activation not in ["relu", "leaky_relu"]:
            raise ValueError("Invalid activation value. Use 'relu' or 'leaky_relu'.")

        if (
            isinstance(self.padding, str)
            and self.padding in ["same"]
            or isinstance(self.padding, (int, tuple))
        ):
            return
        else:
            raise ValueError(
                "Invalid padding value. Use 'same', an integer, or a tuple."
            )

    def build(self, input_channels: int):
        """Construct the convolutional layer followed by the activation.

        Args:
            input_channels (int): Number of input channels.

        Returns:
            nn.Sequential: A sequential model containing the convolutional layer and activation.
        """
        padding = self._calculate_padding()
        return nn.Conv2d(
            in_channels=input_channels,
            out_channels=self.filters,
            kernel_size=self.kernel_size,
            stride=self.strides,
            padding=padding,
        )

    def _calculate_padding(
        self,
    ) -> int | tuple[int, int]:
        """Calculate padding based on the given padding type.

        Args:
            input_channels (int): Number of input channels.

        Returns:
            int | tuple[int, int]: The padding value or tuple.
        """
        if isinstance(self.padding, str) and self.padding == "same":
            # Calculate padding for 'same' effect
            if self.strides == 1:
                padding = (self.kernel_size - 1) // 2
                return (padding, padding)
            else:
                padding = [(self.kernel_size - 1) // 2] * 2
                out_padding = (self.strides - 1) * (self.kernel_size - 1)
                return (
                    (padding[0], padding[1])
                    if self.strides == 1
                    else (padding[0] + out_padding, padding[1] + out_padding)
                )
        elif isinstance(self.padding, (int, tuple)):
            return self.padding
        else:
            raise ValueError(f"Unsupported padding type: {self.padding}")

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
