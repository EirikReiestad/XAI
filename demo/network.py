from rl.src.common import ConvLayer

CONV_LAYERS = [
    ConvLayer(
        filters=32,
        kernel_size=2,
        strides=2,
        activation="relu",
        padding="same",
    ),
    ConvLayer(
        filters=32,
        kernel_size=2,
        strides=2,
        activation="relu",
        padding="same",
    ),
    ConvLayer(
        filters=32,
        kernel_size=2,
        strides=2,
        activation="relu",
        padding="same",
    ),
]
