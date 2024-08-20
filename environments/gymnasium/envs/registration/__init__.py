import gymnasium as gym
from environments.gymnasium.envs.maze import MazeEnv

gym.register(
    id="MazeEnv-v0",
    entry_point="environments/gymnasium/envs/maze/wrapper:CustomMazeWrapper",  # Module path to the wrapper class
    kwargs={"env": MazeEnv()},  # Arguments for the wrapper
)
