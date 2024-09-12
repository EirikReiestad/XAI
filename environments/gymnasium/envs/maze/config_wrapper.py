import gymnasium as gym


class ConfigWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, config: dict):
        super().__init__(env)
        self.env = env
        self.config = config
        self.env.__init__(self.config)


def make_env_with_config(config: dict) -> gym.Env:
    env = gym.make("MazeEnv-v0")
    return ConfigWrapper(env, config)
