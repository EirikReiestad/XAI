import environments.gymnasium.envs.registration
from demo import settings

assert settings.NUM_EPISODES > 0, "NUM_EPISODES must be greater than 0"
assert settings.RENDER_EVERY > 0, "RENDER_EVERY must be greater than 0"
assert settings.SLOWING_FACTOR > 0, "SLOWING_FACTOR must be greater than 0"

assert isinstance(settings.PLOTTING, bool), "PLOTTING must be a boolean"
