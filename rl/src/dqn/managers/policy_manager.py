from rl.src.dqn.policies.dqn_policy import DQNPolicy
import gymnasium as gym


class PolicyManager:
    @staticmethod
    def get_policy(
        policy: str | DQNPolicy,
        observation_space: gym.Space,
        action_space: gym.Space,
        hidden_layers: list[int],
        conv_layers: list[int],
        dueling: bool,
    ) -> DQNPolicy:
        if isinstance(policy, str):
            if policy.lower() == "dqnpolicy":
                return DQNPolicy(
                    observation_space, action_space, hidden_layers, conv_layers, dueling
                )
            raise ValueError(f"Unknown policy: {policy}")
        return policy
