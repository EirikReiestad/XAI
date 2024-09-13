from rl.src.dqn import DQN
from rl.src.dqn.policies import DQNPolicy
from rl.src.base import MultiAgentBase
from rl.src.dqn.components.types import Rollout, RolloutReturn
import torch
from itertools import count
from environments.gymnasium.wrappers import MultiAgentEnv


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentWrapper(MultiAgentBase):
    def __init__(
        self, env: MultiAgentEnv, num_agents: int, dqn_policy: str | DQNPolicy, **kwargs
    ):
        self.env = env
        self.num_agents = num_agents
        self.agents = [DQN(env, dqn_policy, **kwargs) for _ in range(num_agents)]

    def learn(self, total_timesteps: int):
        for _ in range(total_timesteps):
            result = self._collect_rollouts()

    def _collect_rollouts(self) -> list[RolloutReturn]:
        state, info = self.env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        rollout_returns = [RolloutReturn() for _ in range(self.num_agents)]
        episode_rewards = [0.0 for _ in range(self.num_agents)]
        episode_length = 0

        for t in count():
            actions = self.predict(state)
            actions = [action.item() for action in actions]
            (
                observation,
                terminated,
                observations,
                rewards,
                terminals,
                truncated,
                infos,
            ) = self.env.multi_step(actions)

            episode_rewards += rewards
            episode_length += 1

            next_states = [
                torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(
                    0
                )
                for observation in observations
            ]

            for i in range(self.num_agents):
                action = torch.tensor([actions[i]], device=device)
                reward = torch.tensor([rewards[i]], device=device)
                next_state = next_states[i]
                term = torch.tensor([terminals[i]], device=device)
                trunc = torch.tensor([truncated[i]], device=device)
                rollout = Rollout(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    terminated=term,
                    truncated=trunc,
                    value=torch.tensor([0.0], device=device),
                    log_prob=torch.tensor([0.0], device=device),
                    advantage=torch.tensor([0.0], device=device),
                    returns=torch.tensor([0.0], device=device),
                    next_value=torch.tensor([0.0], device=device),
                )
                rollout_returns[i].append(rollout)

                self.agents[i].train(
                    states=[rollout.state],
                    actions=[rollout.action],
                    observations=[rollout.next_state],
                    rewards=[rollout.reward],
                    terminated=[rollout.terminated],
                    truncated=[rollout.truncated],
                )

            state = observation

            if terminated:
                break

        for i in range(self.num_agents):
            self.wandb_manager.log(
                {
                    f"agent{i}_reward": episode_rewards,
                }
            )
        self.wandb_manager.log(
            {
                "episode_length": episode_length,
            }
        )

        return rollout_returns
