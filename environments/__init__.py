from environments import settings
from environments.gymnasium.utils.state_type import StateType

# Check settings

# Maze settings
assert settings.ENV_HEIGHT > 0, "ENV_HEIGHT must be greater than 0"
assert isinstance(settings.ENV_HEIGHT, int), "MAZE_HEIGHT must be an integer"
assert settings.ENV_WIDTH > 0, "MAZE_WIDTH must be greater than 0"
assert isinstance(settings.ENV_WIDTH, int), "MAZE_WIDTH must be an integer"

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

# State
assert settings.STATE_TYPE in StateType, "STATE_TYPE must be a StateType"

# Coop
assert settings.COOP_RADIUS >= 0, "COOP_RADIUS must be greater or equal to 0"
