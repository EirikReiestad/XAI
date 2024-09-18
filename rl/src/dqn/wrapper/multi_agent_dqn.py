from itertools import count

import numpy as np
import torch

from environments.gymnasium.wrappers import MultiAgentEnv
from rl.src.base import MultiAgentBase
from rl.src.dqn import DQN
from rl.src.dqn.components.types import Rollout, RolloutReturn
from rl.src.dqn.policies import DQNPolicy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentDQN(MultiAgentBase):
    def __init__(
        self,
        env: MultiAgentEnv,
        num_agents: int,
        dqn_policy: str | DQNPolicy,
        wandb: bool = False,
        save_every_n_episodes: int = 10,
        **kwargs,
    ):
        super().__init__(wandb=wandb)
        self.env = env
        self.num_agents = num_agents
        self.agents = [DQN(env, dqn_policy, **kwargs) for _ in range(num_agents)]

        self.save_every_n_episodes = save_every_n_episodes

    def learn(self, total_timesteps: int) -> list[list[RolloutReturn]]:
        results = []
        for i in range(total_timesteps):
            rollout, episode_rewards, steps, episode_data = self._collect_rollouts()

            log = dict()
            for agent in range(self.num_agents):
                log[f"agent{agent}_reward"] = episode_rewards[agent]
                log["steps"] = steps
                log["episode"] = i
                for key, value in episode_data[agent].items():
                    log[f"agent{agent}_{key}"] = value
            self.wandb_manager.log(log)
            results.append(rollout)

            if i % self.save_every_n_episodes == 0:
                self.save(append=f"episode_{i}")
        return results

    def _collect_rollouts(
        self,
    ) -> tuple[list[RolloutReturn], list[float], int, list[dict]]:
        state, info = self.env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        rollout_returns = [RolloutReturn() for _ in range(self.num_agents)]
        episode_rewards = [0.0 for _ in range(self.num_agents)]
        episode_length = 0

        data = [{} for _ in range(self.num_agents)]

        for t in count():
            actions = self.predict(state)
            actions = [action.item() for action in actions]
            (
                full_state,
                observation,
                done,
                observations,
                rewards,
                terminals,
                truncated,
                infos,
            ) = self.env.get_wrapper_attr("step_multiple")(actions)

            episode_rewards += rewards
            episode_rewards = [e + r for e, r in zip(episode_rewards, rewards)]
            episode_length += 1

            next_states = [
                torch.tensor(observation, device=device, dtype=torch.float32).unsqueeze(
                    0
                )
                for observation in observations
            ]

            for i in range(self.num_agents):
                skip = infos[i].get("skip")
                if skip:
                    continue

                additional_data = infos[i].get("data")
                if additional_data is not None:
                    for key, value in additional_data.items():
                        data[i][key] = data[i].setdefault(key, 0) + value

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
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            if done:
                break

        return rollout_returns, episode_rewards, episode_length, data

    def predict(self, state: torch.Tensor) -> list[torch.Tensor]:
        return [agent.predict(state) for agent in self.agents]

    def load(self):
        for i in range(self.num_agents):
            self.agents[i].load()

    def save(self, append: str = ""):
        for i in range(self.num_agents):
            self.agents[i].save(
                f"_agent{i}_{append}_", wandb_manager=self.wandb_manager
            )

    def get_q_values(self, states: np.ndarray, agent: int) -> np.ndarray:
        return self.agents[agent].get_q_values(states)
