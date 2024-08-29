import matplotlib.pyplot as plt


def q_value_to_color(
    q_value: float, min_q: float, max_q: float
) -> tuple[int, int, int]:
    """Map a Q-value to a color based on its magnitude."""
    normalized = (q_value - min_q) / (max_q - min_q)  # Normalize Q-value

    colormap = plt.get_cmap("coolwarm")
    rgba_color = colormap(normalized)

    color = tuple(int(rgba_color[i] * 255) for i in range(3))
    if len(color) != 3:
        raise ValueError("Color must have 3 channels.")
    return color
