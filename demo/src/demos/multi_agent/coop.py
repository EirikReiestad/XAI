import logging
from itertools import count
from time import sleep

import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo import network, settings
from demo.src.common.episode_information import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import EnvironmentWrapper
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule

# Register Gym environment
gym.register(
    id="Coop-v0",
    entry_point="environments.gymnasium.envs.coop.coop:CoopEnv",
)


class Demo:
    """Class for running the Coop demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Demo class with settings and plotter."""
        self.env_wrapper = EnvironmentWrapper(env_id="Coop-v0")

        self.num_agents = self.env_wrapper.num_agents

        self.episode_informations = []
        for _ in range(self.num_agents):
            self.episode_informations.append(EpisodeInformation([], []))

        self.plotter = Plotter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ipython = "inline" in matplotlib.get_backend()

    def run(self):
        """Run the demo, interacting with the environment and training the DQN."""
        state, info = self.env_wrapper.reset()
        n_actions = self.env_wrapper.action_space.n
        conv_layers = self._get_conv_layers(info)

        dqn = DQNModule(state.shape, n_actions, conv_layers=conv_layers)
        self.dqns = [dqn] * self.num_agents

        plt.ion()

        try:
            for i_episode in range(settings.NUM_EPISODES):
                self._run_episode(i_episode, state, info)
        except Exception as e:
            logging.exception(e)
        finally:
            self.env_wrapper.close()
            logging.info("Complete")
            self.plotter.update(self.episode_informations, show_result=True)
            plt.ioff()
            plt.show()

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Handle the episode by interacting with the environment and training the DQN."""
        state, _ = self.env_wrapper.reset()
        total_rewards = np.zeros(self.num_agents)

        dones = [False] * self.num_agents

        for t in count():
            if i_episode % settings.RENDER_EVERY == 0:
                self.env_wrapper.render()

            _, rewards, dones, full_states = self._run_step(state)
            total_rewards += rewards

            done = any(dones)
            if not done:
                full_state, reward, done = self.env_wrapper.concat_state(full_states)
                state = self.env_wrapper.update_state(full_state[0].numpy())
                for agent in range(self.num_agents):
                    total_rewards[agent] += reward

            if done:
                for agent in range(self.num_agents):
                    self.episode_informations[agent].durations.append(t + 1)
                    self.episode_informations[agent].rewards.append(
                        total_rewards[agent]
                    )
                self.plotter.update(self.episode_informations)
                break

    def _run_step(
        self, state: torch.Tensor
    ) -> tuple[list[torch.Tensor], list[float], list[bool], list[np.ndarray]]:
        """Run a step in the environment and train the DQN."""
        total_reward = [0.0] * self.num_agents
        new_states = [torch.empty(0)] * self.num_agents
        full_states = []
        dones = [False] * self.num_agents

        for agent in range(self.num_agents):
            action = self.dqns[agent].select_action(state)

            self.env_wrapper.set_active_agent(agent)
            observation, reward, terminated, truncated, info = self.env_wrapper.step(
                action.item()
            )

            full_state = info.get("full_state")
            if full_state is None:
                raise ValueError("Full state must be returned from the environment.")

            full_states.append(full_state)

            reward = float(reward)
            total_reward[agent] += reward

            done, new_state = self.dqns[agent].train(
                state,
                action,
                observation,
                reward,
                terminated,
                truncated,
            )

            if new_state is not None:
                new_states[agent] = new_state
            dones[agent] = done

        if any(new_state is None for new_state in new_states):
            raise ValueError("New state must be returned from the DQN.")

        return new_states, total_reward, dones, full_states

    def _get_conv_layers(self, info) -> list[ConvLayer]:
        """Create convolutional layers based on the state type."""
        state_type = info.get("state_type") if info else None
        if state_type in {"rgb", "full"}:
            network.CONV_LAYERS
        return []


if __name__ == "__main__":
    demo = Demo()
    demo.run()
