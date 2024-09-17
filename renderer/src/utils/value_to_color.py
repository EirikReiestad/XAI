import matplotlib.pyplot as plt


def value_to_color(value: float, min: float, max: float) -> tuple[int, int, int]:
    """Map a Q-value to a color based on its magnitude."""
    if max - min == 0:
        normalized = 0
    else:
        normalized = (value - min) / (max - min)  # Normalize Q-value

    colormap = plt.get_cmap("coolwarm")
    rgba_color = colormap(normalized)

    color = tuple(int(rgba_color[i] * 255) for i in range(3))
    if len(color) != 3:
        raise ValueError("Color must have 3 channels.")
    return color
