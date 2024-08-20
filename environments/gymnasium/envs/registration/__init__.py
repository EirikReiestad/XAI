import gymnasium as gym

from environments.gymnasium.envs.coop import CoopEnv
from environments.gymnasium.envs.maze import MazeEnv

gym.register(
    id="MazeEnv-v0",
    entry_point="environments.gymnasium.envs.maze.wrapper:MazeEnvWrapper",
    kwargs={"env": MazeEnv()},
)

gym.register(
    id="CoopEnv-v0",
    entry_point="environments.gymnasium.envs/coop.wrapper:CoopEnvWrapper",
    kwargs={"env": CoopEnv()},
)
