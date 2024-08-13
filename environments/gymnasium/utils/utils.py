import random
from typing import List, Optional

from environments.gymnasium.utils.position import Position


def generate_random_position(
    width: int, height: int, exclude_positions: Optional[List[Position]] = None
) -> Position:
    """
    Generate a random position within the specified width and height,
    ensuring the position is not in the list of excluded positions.

    Args:
        width (int): The width of the area.
        height (int): The height of the area.
        exclude_positions (Optional[List[Position]]): A list of positions to exclude from the random generation.

    Returns:
        Position: A randomly generated position not in the excluded list.

    Raises:
        ValueError: If a unique position cannot be found.
    """
    exclude_positions = exclude_positions or []
    max_attempts = width * height

    for _ in range(max_attempts):
        position = Position(random.randint(0, width - 1), random.randint(0, height - 1))
        if position not in exclude_positions:
            return position

    raise ValueError("Could not generate a unique random position.")
