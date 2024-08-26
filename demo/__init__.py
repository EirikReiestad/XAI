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
assert isinstance(settings.SAVE_MODEL_NAME, str), "SAVE_MODEL_NAME must be a string"
assert settings.SAVE_MODEL_NAME != "", "MODEL_NAME cannot be empty"
assert isinstance(settings.LOAD_MODEL_NAME, str), "LOAD_MODEL_NAME must be a string"
assert settings.LOAD_MODEL_NAME != "", "LOAD_MODEL_NAME cannot be empty"

if settings.PRETRAINED:
    if settings.DEMO == DemoType.MAZE:
        assert os.path.exists(
            os.path.join("history", "models", "maze", settings.LOAD_MODEL_NAME + ".pt")
        ), "Model file does not exist"
    elif settings.DEMO == DemoType.COOP:
        assert os.path.exists(
            os.path.join(
                "history", "models", "coop", settings.LOAD_MODEL_NAME + "_agent0.pt"
            )
        ), "Model file does not exist"

__all__ = ["DemoType", "settings"]
