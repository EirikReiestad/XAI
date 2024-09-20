from demo import settings
from demo.src.demos import DemoType, RLType
from environments.gymnasium.envs import registration
import logging

logging.basicConfig(filename="log.txt", level=logging.INFO)

__all__ = ["DemoType", "settings", "RLType"]
