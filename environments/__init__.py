from environments.gymnasium.utils.enums import StateType
from environments import settings

# Check settings

# Maze settings
assert settings.MAZE_HEIGHT > 0, "MAZE_HEIGHT must be greater than 0"
assert isinstance(settings.MAZE_HEIGHT, int), "MAZE_HEIGHT must be an integer"
assert settings.MAZE_WIDTH > 0, "MAZE_WIDTH must be greater than 0"
assert isinstance(settings.MAZE_WIDTH, int), "MAZE_WIDTH must be an integer"

# Reward settings
rewards = [
    ("GOAL_REWARD", settings.GOAL_REWARD),
    ("MOVE_REWARD", settings.MOVE_REWARD),
    ("TERMINATED_REWARD", settings.TERMINATED_REWARD),
    ("TRUNCATED_REWARD", settings.TRUNCATED_REWARD),
]

for reward in rewards:
    if type(reward[1]) is not int and type(reward[1]) is not float:
        raise ValueError(f"{reward[0]} must be an integer or a float")

# Screen settings
assert settings.SCREEN_WIDTH > 0, "SCREEN_WIDTH must be greater than 0"
assert isinstance(settings.SCREEN_WIDTH, int), "SCREEN_WIDTH must be an integer"
assert settings.SCREEN_HEIGHT > 0, "SCREEN_HEIGHT must be greater than 0"
assert isinstance(settings.SCREEN_HEIGHT, int), "SCREEN_HEIGHT must be an integer"

# Path: environments
assert isinstance(settings.FILENAME, str), "FILENAME must be a string"
