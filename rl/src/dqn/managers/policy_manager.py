from rl.src.dqn.policies.dqn_policy import DQNPolicy


class PolicyManager:
    def __init__(self, policy: str | DQNPolicy, **kwargs):
        self._policy = self._get_policy(policy, **kwargs)

    def _get_policy(self, policy: str | DQNPolicy, **kwargs) -> DQNPolicy:
        if isinstance(policy, str):
            if policy.lower() == "dqnpolicy":
                observation_space = kwargs.get("observation_space")
                action_space = kwargs.get("action_space")
                if observation_space is None or action_space is None:
                    raise ValueError(
                        "observation_space and action_space must be provided when policy is a string"
                    )
                return DQNPolicy(observation_space, action_space)
            raise ValueError(f"Unknown policy: {policy}")
        return policy

    @property
    def policy(self) -> DQNPolicy:
        return self._policy
