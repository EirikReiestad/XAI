"""DQN Module.
This module contains the DQN agent that interacts with the environment.
"""

import json
import math
import random
from itertools import count

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

from rl.src.base import BaseRL
from rl.src.common import checker, setter
from rl.src.managers import WandBConfig, WandBManager

from .common.hyperparameter import DQNHyperparameter
from .components.types import Rollout, RolloutReturn, Transition
from .managers import MemoryManager, OptimizerManager
from .policies import DQNPolicy, QNetwork, get_policy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DQN(BaseRL):
    """DQN Module for managing the agent, including training and evaluation."""

    policy: DQNPolicy
    policy_net: QNetwork
    target_net: QNetwork

    def __init__(
        self,
        policy: str | DQNPolicy,
        env: gym.Env,
        seed: int | None = None,
        dueling: bool = False,
        double: bool = False,
        memory_size: int = 5000,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 0.9,
        epsilon_end: float = 0.05,
        epsilon_decay: int = 1000,
        batch_size: int = 128,
        tau: float = 0.005,
        wandb: bool = False,
    ) -> None:
        setter.set_seed(seed)

        self.env = env
        self.n_actions = env.action_space.n

        self.hp = DQNHyperparameter(
            lr, gamma, epsilon_start, epsilon_end, epsilon_decay, batch_size, tau
        )

        self.policy = get_policy(
            policy,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
        )
        self.policy_net = self.policy.policy_net
        self.target_net = self.policy.target_net

        optimizer = OptimizerManager(self.policy_net, self.hp.lr)
        self.optimizer = optimizer.initialize()

        self.memory = MemoryManager(memory_size).initialize()

        wandb_config = WandBConfig(
            project="dqn",
            run_name="default",
            tags=[],
            other={},
        )
        self.wandb_manager = WandBManager(wandb, wandb_config)

        self.double = double
        self.step_count = 0
        self.steps_done = 0

    def learn(
        self,
        total_timesteps: int,
    ):
        while self.steps_done < total_timesteps:
            result = self._collect_rollout()

            self.train(
                states=result.states,
                actions=result.actions,
                observations=result.next_states,
                rewards=result.rewards,
                terminated=result.terminals,
                truncated=result.truncated,
            )

    def _collect_rollout(self) -> RolloutReturn:
        state, info = self.env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)
        rewards = 0
        episode_length = 0

        rollout_return = RolloutReturn()

        for t in count():
            action = self.predict(state)
            observation, reward, terminated, truncated, info = self.env.step(
                action.item()
            )
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

            state = next_state

            if terminated or truncated:
                episode_length = t + 1
                break

        self.wandb_manager.log(
            {
                "reward": rewards,
                "episode_length": episode_length,
            }
        )

        return rollout_return

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

    def predict(self, state: torch.Tensor) -> torch.Tensor:
        checker.raise_if_not_same_shape_as_observation(
            state, self.env.observation_space, "state"
        )

        eps_threshold = self.hp.eps_end + (
            self.hp.eps_start - self.hp.eps_end
        ) * math.exp(-self.steps_done / self.hp.eps_decay)
        self.steps_done += 1

        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[random.randrange(self.n_actions)]],
                device=device,
                dtype=torch.long,
            )

    def _optimize_model(self) -> None:
        """Perform one optimization step on the policy network."""
        if not self._can_optimize():
            return

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
                    raise ValueError("All states must be PyTorch tensors.")

        q_values = np.ndarray((*states.shape, self.n_actions), dtype=np.float32)
        height = states.shape[0]
        width = states.shape[1]
        for y in range(height):
            for x in range(width):
                with torch.no_grad():
                    q_values[y, x] = self.policy_net(states[y, x]).cpu()
        return q_values

    def get_q_values_map(self, states: np.ndarray, **args) -> np.ndarray:
        q_values = self.get_q_values(states)

        if states.shape[:2] != q_values.shape[:2]:
            raise ValueError(
                f"States shape {states.shape[:2]} does not match Q-values shape {q_values.shape[:2]}"
            )

        if args.get("max_q_values"):
            max_q_values = np.max(q_values, axis=2)
            normalized_max_q_values = (max_q_values - np.min(max_q_values)) / np.ptp(
                max_q_values
            )
            return normalized_max_q_values

        adjusted_q_values = q_values + np.abs(np.min(q_values))
        normalized_q_values = (adjusted_q_values - np.min(adjusted_q_values)) / np.ptp(
            adjusted_q_values
        )
        cumulated_q_values = np.zeros(
            (states.shape[0], states.shape[1]), dtype=np.float32
        )
        cumulated_q_values_count = np.zeros(
            (states.shape[0], states.shape[1]), dtype=np.int32
        )

        width, height = states.shape[:2]
        for y in range(height):
            for x in range(width):
                # We use the following order: up, down, left, right
                if x > 0:
                    cumulated_q_values[y, x - 1] += normalized_q_values[y, x, 2]
                    cumulated_q_values_count[y, x - 1] += 1
                if x < height - 1:
                    cumulated_q_values[y, x + 1] += normalized_q_values[y, x, 3]
                    cumulated_q_values_count[y, x + 1] += 1
                if y > 0:
                    cumulated_q_values[y - 1, x] += normalized_q_values[y, x, 0]
                    cumulated_q_values_count[y - 1, x] += 1
                if y < width - 1:
                    cumulated_q_values[y + 1, x] += normalized_q_values[y, x, 1]
                    cumulated_q_values_count[y + 1, x] += 1

        for y in range(height):
            for x in range(width):
                if cumulated_q_values_count[y, x] > 0:
                    cumulated_q_values[y, x] /= cumulated_q_values_count[y, x]

        normalized_cumulated_q_values = (
            cumulated_q_values - np.min(cumulated_q_values)
        ) / np.ptp(cumulated_q_values)

        return normalized_cumulated_q_values

    def save(self, path: str) -> None:
        """Save the policy network to the specified path."""
        if not path.endswith(".pt"):
            path += ".pt"
        torch.save(self.policy_net.state_dict(), path)

        meta_data = {
            "steps_done": self.steps_done,
            "step_count": self.step_count,
        }
        meta_data_path = path.replace(".pt", "_meta_data.json")
        json.dump(meta_data, open(meta_data_path, "w"))

    def load(self, path: str) -> None:
        """Load the policy network from the specified path."""
        if not path.endswith(".pt"):
            path += ".pt"
        self.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.policy_net.eval()
        self.target_net.eval()

        meta_data_path = path.replace(".pt", "_meta_data.json")
        meta_data = json.load(open(meta_data_path, "r"))
        self.steps_done = meta_data["steps_done"]
        self.step_count = meta_data["step_count"]
