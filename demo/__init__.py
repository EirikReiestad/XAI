from demo import settings
from demo.src.demos import DemoType, RLType
from environments.gymnasium.envs import registration

assert settings.EPOCHS > 0, "NUM_EPISODES must be greater than 0"

__all__ = ["DemoType", "settings", "RLType"]
