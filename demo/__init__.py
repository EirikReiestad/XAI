import environments.gymnasium.envs.registration
import os
from demo import settings
from demo.src.demos.demo_type import DemoType

assert settings.NUM_EPISODES > 0, "NUM_EPISODES must be greater than 0"
assert settings.RENDER_EVERY > 0, "RENDER_EVERY must be greater than 0"
assert settings.SLOWING_FACTOR > 0, "SLOWING_FACTOR must be greater than 0"

assert isinstance(settings.PLOTTING, bool), "PLOTTING must be a boolean"

assert isinstance(settings.SAVE_MODEL, bool), "SAVE_MODEL must be a boolean"
assert settings.SAVE_EVERY > 0, "SAVE_EVERY must be greater than 0"

assert isinstance(settings.PRETRAINED, bool), "USE_MODEL must be a boolean"
assert isinstance(settings.MODEL_NAME, str), "MODEL_NAME must be a string"
assert settings.MODEL_NAME != "", "MODEL_NAME cannot be empty"

if settings.PRETRAINED:
    assert os.path.exists(
        os.path.join("models", "models", settings.MODEL_NAME)
    ), "Model file does not exist"

__all__ = ["DemoType", "settings"]
