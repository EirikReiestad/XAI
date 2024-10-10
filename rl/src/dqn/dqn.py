"""DQN Module.
This module contains the DQN agent that interacts with the environment.
"""

import logging
import math
import os
import random
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from rl.src.base import SingleAgentBase
from rl.src.common import checker, setter
from rl.src.managers import WandBConfig, WandBManager

from .common.hyperparameter import DQNHyperparameter
from .components.types import Rollout, RolloutReturn, Transition
from .managers import MemoryManager, OptimizerManager, PolicyManager
from .policies import DQNPolicy, QNetwork

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(SingleAgentBase):
    """DQN Module for managing the agent, including training and evaluation."""

    policy: DQNPolicy
    policy_net: QNetwork
    target_net: QNetwork

    def __init__(
        self,
        env: gym.Env,
        policy: str | DQNPolicy,
        seed: int | None = None,
        agent_id: int = 0,
        dueling: bool = False,
        double: bool = True,
        memory_size: int = 10000,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 50000,
        batch_size: int = 64,
        tau: float = 0.005,
        wandb: bool = False,
        wandb_config: WandBConfig | None = None,
        save_model: bool = False,
        save_every_n_episodes: int = 100,
        model_path: str = "models/",
        model_name: str = "dqn",
        load_model: bool = False,
        run_path: str = "",
        model_artifact: str = "",
        version_number: str = "latest",
        gif: bool = False,
        gif_path="assets/gifs",
        gif_name="dqn",
    ) -> None:
        super().__init__(wandb, wandb_config)

        setter.set_seed(seed)

        self.env = env
        self.n_actions = env.action_space.n
        self.agent_id = agent_id

        self.save_model = save_model
        self.model_path = model_path
        self.model_name = model_name
        self.save_every_n_episodes = save_every_n_episodes

        self.hp = DQNHyperparameter(
            lr, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, tau
        )
        self.eps_threshold = 0

        self.policy = PolicyManager(
            policy,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        ).policy

        self.policy_net = self.policy.policy_net
        self.target_net = self.policy.target_net

        if load_model:
            self.load(run_path, model_artifact, version_number)

        optimizer = OptimizerManager(self.policy_net, self.hp.lr)
        self.optimizer = optimizer.initialize()

        self.memory = MemoryManager(memory_size).initialize()

        self.double = double
        self.steps_done = 0
        self.episodes = 0

        self._init_gif(gif, gif_path, gif_name)

    def _init_gif(self, gif: bool, gif_path: str, gif_name: str) -> None:
        self.gif = gif
        if not os.path.exists(gif_path):
            logging.warning(f"Directory {gif_path} does not exist.")
            self.gif = False
        elif self.gif and not self.wandb_manager.active:
            logging.warning("GIF is enabled but Weights & Biases is not enabled.")
            self.gif = False
        elif self.gif and self.env.render_mode != "rgb_array":
            logging.warning(
                "GIF is enabled but the environment does not support RGB array rendering."
            )
            self.gif = False
        else:
            self.gif_samples = 10
            self.gif_path = gif_path
            self.gif_name = gif_name

    def learn(self, total_timesteps: int):
        self.policy_net.train()
        self.target_net.eval()
        max_gif_reward = -float("inf")

        max_gif_reward = -np.inf
        frames = []

        try:
            for _ in range(total_timesteps):
                self.episodes += 1
                _, episode_rewards, steps, gif_frames = self._collect_rollout()

                if (
                    self.gif
                    and len(gif_frames) > 0
                    and episode_rewards > max_gif_reward
                ):
                    max_gif_reward = episode_rewards
                    frames = gif_frames

                self.wandb_manager.log(
                    {
                        "reward": episode_rewards,
                        "episode_length": steps,
                        "steps_done": self.steps_done,
                        "epsilon_threshold": self.eps_threshold,
                    },
                )
                if self.episodes % self.save_every_n_episodes == 0:
                    max_gif_reward = -float("inf")
                    self.save_gif_local(frames)
                    self.save(self.episodes)
        except Exception as e:
            logging.error(f"Error: {e}")
            self.close()
        finally:
            self.close()

    def _collect_rollout(
        self,
    ) -> tuple[
        RolloutReturn,
        float,
        int,
        list,
    ]:
        state, _ = self.env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        rewards = 0
        episode_length = 0

        rollout_return = RolloutReturn()

        frames = []

        for t in count():
            action = self.predict_action(state)
            observation, reward, terminated, truncated, info = self.env.step(
                action.item()
            )
            skip = info.get("skip")
            if skip:
                continue

            next_state = torch.tensor(
                observation, device=device, dtype=torch.float32
            ).unsqueeze(0)
            rewards += float(reward)

            rollout = Rollout(
                state=state,
                action=action,
                reward=torch.tensor([reward], device=device),
                next_state=next_state,
                terminated=torch.tensor([terminated], device=device),
                truncated=torch.tensor([truncated], device=device),
                value=torch.tensor([0.0], device=device),
                log_prob=torch.tensor([0.0], device=device),
                advantage=torch.tensor([0.0], device=device),
                returns=torch.tensor([0.0], device=device),
                next_value=torch.tensor([0.0], device=device),
            )
            rollout_return.append(rollout)

            self.train(
                states=[rollout.state],
                actions=[rollout.action],
                observations=[rollout.next_state],
                rewards=[rollout.reward],
                terminated=[rollout.terminated],
                truncated=[rollout.truncated],
            )

            state = next_state

            if self.gif:
                if (
                    self.save_every_n_episodes - self.gif_samples
                    < self.steps_done % self.save_every_n_episodes
                    < self.save_every_n_episodes
                ):
                    rgb_array = self.env.render()
                    assert isinstance(rgb_array, np.ndarray)
                    pil_image = Image.fromarray(rgb_array.transpose(1, 0, 2), "RGB")
                    pil_image = pil_image.rotate(-90, expand=True)
                    frames.append(pil_image)

            if terminated or truncated:
                episode_length = t + 1
                break

        return rollout_return, rewards, episode_length, frames

    def train(
        self,
        states: list[torch.Tensor],
        actions: list[torch.Tensor],
        observations: list[torch.Tensor],
        rewards: list[torch.Tensor],
        terminated: list[torch.Tensor],
        truncated: list[torch.Tensor],
    ):
        """Store transition and optimize the model."""
        checker.raise_if_not_all_same_shape_as_observation(
            states, self.env.observation_space, "states"
        )
        checker.raise_if_not_all_same_shape_as_observation(
            observations, self.env.observation_space, "observations"
        )

        next_states = [
            obs.clone().detach() if not (term or trunc) else None
            for obs, term, trunc in zip(observations, terminated, truncated)
        ]

        for state, action, next_state, reward in zip(
            states, actions, next_states, rewards
        ):
            self.memory.push(state, action, next_state, reward)

        self._optimize_model()

        self._soft_update_target_net()

    def predict(self, states: torch.Tensor) -> list[np.ndarray] | np.ndarray:
        for state in states:
            checker.raise_if_not_same_shape_as_observation(
                state, self.env.observation_space, "state"
            )

        with torch.no_grad():
            return self.policy_net(states).detach().cpu().numpy()

    def predict_action(self, state: torch.Tensor) -> torch.Tensor:
        checker.raise_if_not_same_shape_as_observation(
            state, self.env.observation_space, "state"
        )

        self.eps_threshold = self.hp.eps_end + (
            self.hp.eps_start - self.hp.eps_end
        ) * math.exp(-self.steps_done / self.hp.eps_decay)
        self.steps_done += 1

        if random.random() > self.eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device,
                dtype=torch.long,
            )

    def model(self) -> torch.nn.Module:
        return self.policy_net

    def _optimize_model(self) -> None:
        """Perform one optimization step on the policy network."""
        if not self._can_optimize():
            return

        device = next(self.policy_net.parameters()).device

        transitions, indices, weights = self.memory.sample(self.hp.batch_size)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if action_batch.dim() == 1:
            action_batch = action_batch.unsqueeze(1)

        state_batch = state_batch.to(device)
        action_batch = action_batch.to(device)
        reward_batch = reward_batch.to(device)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)
        next_state_values = torch.zeros(self.hp.batch_size, device=device)

        if self.double:
            with torch.no_grad():
                next_actions = (
                    self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1)
                )
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states)
                    .gather(1, next_actions)
                    .squeeze(1)
                )
        else:
            with torch.no_grad():
                next_state_values[non_final_mask] = (
                    self.target_net(non_final_next_states).max(1).values
                )

        expected_state_action_values = next_state_values * self.hp.gamma + reward_batch
        td_errors = expected_state_action_values.unsqueeze(1) - state_action_values
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
        loss = loss * torch.tensor(
            weights, device=device, dtype=torch.float32
        ).unsqueeze(1)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors.squeeze(1).detach().numpy())

    def _can_optimize(self) -> bool:
        if len(self.memory) < self.hp.batch_size:
            return False
        return True

    def _soft_update_target_net(self) -> None:
        """Soft update the target network parameters."""
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[
                key
            ] * self.hp.tau + target_net_state_dict[key] * (1 - self.hp.tau)
        self.target_net.load_state_dict(target_net_state_dict)

    def get_q_values(self, states: np.ndarray) -> np.ndarray:
        """Calculate the Q-values for each action in the environment.
        Returns:
            np.ndarray: Array of Q-values for each action.
        """
        for row_state in states:
            for state in row_state:
                if type(state) is not torch.Tensor:
                    raise ValueError(
                        f"All states must be PyTorch tensors, not {type(state)}"
                    )

        q_values = np.ndarray((*states.shape, self.n_actions), dtype=np.float32)
        height = states.shape[0]
        width = states.shape[1]
        for y in range(height):
            for x in range(width):
                state = states[y, x].unsqueeze(0)
                with torch.no_grad():
                    q_values[y, x] = self.policy_net(state).cpu()
        return q_values

    def save(
        self, episode: int, wandb_manager: WandBManager | None = None, append: str = ""
    ) -> None:
        """Save the policy network to the specified path."""
        path = f"{self.model_path}{self.model_name}"
        if not path.endswith(".pt"):
            path += ".pt"
        self._save_model(path, episode, wandb_manager)
        self._delete_model_local(path)
        self._save_gif(episode, wandb_manager, append=append)
        self._delete_gif_local(path)

        if wandb_manager is not None:
            wandb_manager.cleanup()
        else:
            self.wandb_manager.cleanup()

    def _save_model(
        self, path: str, episode: int, wandb_manager: WandBManager | None = None
    ) -> None:
        if not self.save_model:
            return
        torch.save(self.policy_net.state_dict(), path)
        metadata = {"steps_done": self.steps_done}
        if wandb_manager is not None:
            wandb_manager.save_model(path, step=episode, metadata=metadata)
        else:
            self.wandb_manager.save_model(path, step=episode, metadata=metadata)

    def _delete_model_local(self, path: str) -> None:
        if not self.save_model:
            return
        if os.path.exists(path):
            os.remove(path)

    def _delete_gif_local(self, path: str) -> None:
        if not self.gif:
            return
        if os.path.exists(path):
            os.remove(path)

    def _save_gif(
        self, episode: int, wandb_manager: WandBManager | None = None, append: str = ""
    ):
        path = f"{self.gif_path}/{self.gif_name}{append}.gif"
        if wandb_manager is not None:
            wandb_manager.save_gif(path, step=episode, append=append)
        else:
            self.wandb_manager.save_gif(path, step=episode, append=append)

    def save_gif_local(self, frames: list, append: str = "") -> None:
        if len(frames) != 0:
            frames[0].save(
                f"{self.gif_path}/{self.gif_name}{append}.gif",
                save_all=True,
                append_images=frames[1:],
                loop=0,
            )
        else:
            logging.warning("No frames in gif to save.")

    def load(
        self,
        run_path: str,
        model_artifact: str,
        version_number: str,
        wandb_manager: WandBManager | None = None,
    ) -> None:
        """Load the policy network from the specified path."""
        if run_path == "" or model_artifact == "" or version_number == "":
            logging.warning(
                "Run ID, model artifact, and version number must be specified."
            )
        self._load_model(run_path, model_artifact, version_number, wandb_manager)

    def _load_model(
        self,
        run_id: str,
        model_artifact: str,
        version_number: str,
        wandb_manager: WandBManager | None = None,
    ) -> None:
        if wandb_manager is not None:
            artifact_dir, metadata = wandb_manager.load_model(
                run_id, model_artifact, version_number
            )
        else:
            artifact_dir, metadata = self.wandb_manager.load_model(
                run_id, model_artifact, version_number
            )
        if artifact_dir is None or metadata is None:
            return

        path = f"{artifact_dir}/{self.model_name}"
        if not path.endswith(".pt"):
            path += ".pt"

        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

        self.steps_done = metadata["steps_done"]
