import gymnasium as gym

gym.register(
    id="MazeEnv-v0",
    entry_point="environments.gymnasium.envs.maze.maze:MazeEnv",
)

gym.register(
    id="CoopEnv-v0",
    entry_point="environments.gymnasium.envs.coop.coop:CoopEnv",
)
