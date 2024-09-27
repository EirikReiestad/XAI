from demo import settings
from demo.src.demos import DemoType, RLType
from environments.gymnasium.envs import registration
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler("log.txt"),
        logging.StreamHandler(),  # This sends logs to the terminal
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)


__all__ = ["DemoType", "settings", "RLType"]
