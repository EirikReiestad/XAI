import gymnasium as gym


class MetadataWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)

    def feature_names(self) -> list[str]:
        return self.env.get_wrapper_attr("feature_names")()
