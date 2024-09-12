from rl.src.common.policies import BasePolicy
from gymnasium import spaces


class DQNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        super(DQNPolicy, self).__init__(observation_space, action_space)

    def forward(self, x):
        return x
