import logging
import traceback
from itertools import count
from typing import Any

import numpy as np
import torch
from PIL import Image

from environments.gymnasium.wrappers import MultiAgentEnv
from rl.src.base import MultiAgentBase
from rl.src.dqn import DQN
from rl.src.dqn.components.types import Rollout, RolloutReturn
from rl.src.dqn.policies import DQNPolicy
from rl.src.managers import WandBConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiAgentDQN(MultiAgentBase):
    def __init__(
        self,
        env: MultiAgentEnv,
        num_agents: int,
        dqn_policy: str | DQNPolicy,
        wandb: bool = False,
        wandb_config: WandBConfig | None = None,
        save_every_n_episodes: int = 100,
        load_model: bool = False,
        run_path: str = "",
        model_artifact: str = "",
        version_numbers: list[str] = ["v0", "v1"],
        gif: bool = False,
        **kwargs,
    ):
        super().__init__(wandb=wandb, wandb_config=wandb_config)
        self.env = env
        self.num_agents = num_agents
        self.agents = [
            DQN(env, dqn_policy, agent_id=i, **kwargs) for i in range(num_agents)
        ]

        self.save_every_n_episodes = save_every_n_episodes

        self.episodes = 0

        if load_model:
            self.load(run_path, model_artifact, version_numbers)

        self._init_gif(gif)

    def _init_gif(self, gif: bool) -> None:
        self.gif = gif
        if self.gif and self.env.render_mode != "rgb_array":
            logging.warning(
                "GIF is enabled but the environment does not support RGB array rendering."
            )
            self.gif = False
        else:
            self.gif_samples = 10

    def learn(self, total_timesteps: int) -> list[list[RolloutReturn]]:
        results = []
        for agent in self.agents:
            agent.policy_net.train()
            agent.target_net.train()

        max_gif_rewards = [-np.inf for _ in range(self.num_agents)]
        gifs = [[] for _ in range(self.num_agents)]

        total_rewards = (0, 0)

        try:
            for _ in range(total_timesteps):
                self.episodes += 1
                rollout, episode_rewards, steps, info, episode_data, gif = (
                    self._collect_rollouts()
                )

                total_rewards = tuple(
                    sum(x) for x in zip(total_rewards, tuple(episode_rewards))
                )

                for agent in range(self.num_agents):
                    if self.gif:
                        if len(gif) == 0:
                            continue
                        if episode_rewards[agent] > max_gif_rewards[agent]:
                            max_gif_rewards[agent] = episode_rewards[agent]
                            gifs[agent] = gif

                log = dict()
                for agent in range(self.num_agents):
                    log[f"agent{agent}_reward"] = episode_rewards[agent]
                    log["steps_done"] = self.agents[agent].steps_done
                    log[f"agent{agent}_episode_steps"] = steps
                    log["epsilon threshold"] = self.agents[agent].eps_threshold
                    for key, value in episode_data[agent].items():
                        log[f"agent{agent}_{key}"] = value
                    log[f"agent{agent}_reward_per_step"] = (
                        episode_rewards[agent] / steps
                    )
                log["episode"] = self.episodes
                for key, value in info.items():
                    log[key] = value
                self.wandb_manager.log(log)
                results.append(rollout)

                if self.episodes % self.save_every_n_episodes == 0:
                    max_gif_rewards = [-np.inf for _ in range(self.num_agents)]
                    self._save_gifs_local(gifs)
                    self.save(self.episodes)
        except Exception as e:
            logging.error(e)
            logging.error(traceback.format_exc())
            self.close()
        finally:
            self.close()
            return results

    def _collect_rollouts(
        self,
    ) -> tuple[list[RolloutReturn], list[float], int, dict[str, Any], list[dict], list]:
        state, info = self.env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        rollout_returns = [RolloutReturn() for _ in range(self.num_agents)]
        episode_rewards = [0.0 for _ in range(self.num_agents)]
        episode_length = 0

        data = [{} for _ in range(self.num_agents)]

        frames = []

        for t in count():
            actions = self.predict_actions(state)
            actions = [action.item() for action in actions]
            (
                full_state,
                observation,
                done,
                info,
                observations,
                rewards,
                terminals,
                truncated,
                infos,
            ) = self.env.get_wrapper_attr("step_multiple")(actions)

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

            if self.gif:
                if (
                    self.save_every_n_episodes - self.gif_samples
                    < self.episodes % self.save_every_n_episodes
                    < self.save_every_n_episodes
                ):
                    rgb_array = self.env.render()
                    assert isinstance(rgb_array, np.ndarray)
                    rgb_array_transposed = rgb_array.transpose(1, 0, 2)
                    pil_image = Image.fromarray(rgb_array_transposed, "RGB")
                    frames.append(pil_image)

            state = observation
            state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

            if done:
                break

        return rollout_returns, episode_rewards, episode_length, info, data, frames

    def predict(self, state: torch.Tensor) -> list[np.ndarray | list[np.ndarray]]:
        return [agent.predict(state) for agent in self.agents]

    def predict_actions(self, state: torch.Tensor) -> list[torch.Tensor]:
        return [agent.predict_action(state) for agent in self.agents]

    def load(
        self,
        run_id: str,
        model_artifact: str,
        version_numbers: list[str],
    ):
        for i in range(self.num_agents):
            self.agents[i].load(
                run_id,
                model_artifact,
                version_numbers[i],
                wandb_manager=self.wandb_manager,
            )

    def save(self, episode: int):
        for i in range(self.num_agents):
            self.agents[i].save(
                episode, wandb_manager=self.wandb_manager, append=f"_agent{i}"
            )

    def _save_gifs_local(self, gifs: list):
        for i in range(self.num_agents):
            self.agents[i].save_gif_local(gifs[i], append=f"_agent{i}")

    def get_q_values(self, states: np.ndarray, agent: int) -> np.ndarray:
        return self.agents[agent].get_q_values(states)

    @property
    def models(self) -> list[DQN]:
        return self.agents
