from environments import settings
from environments.gymnasium.utils.state_type import StateType

# Check settings

# Maze settings
# Path: environments
assert isinstance(settings.FILENAME, str), "FILENAME must be a string"

assert isinstance(
    settings.RANDOM_SEEKER_POSITION, bool
), "RANDOM_SEEKER_POSITION must be a boolean"
assert isinstance(
    settings.RANDOM_HIDER_POSITION, bool
), "RANDOM_HIDER_POSITION must be a boolean"

# Coop
assert settings.COOP_RADIUS >= 0, "COOP_RADIUS must be greater than 0"
