from utils import Color


def interpolate_color(
    color1: Color, color2: Color, weight: float
) -> tuple[int, int, int]:
    """Interpolate between two colors."""
    r = int(color1.value[0] * (1 - weight) + color2.value[0] * weight)
    g = int(color1.value[1] * (1 - weight) + color2.value[1] * weight)
    b = int(color1.value[2] * (1 - weight) + color2.value[2] * weight)
    return r, g, b
