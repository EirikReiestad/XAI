from abc import ABC, abstractmethod
import logging

import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import settings
from demo.src.common.episode_information import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import MultiAgentEnvironmentWrapper
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule


class BaseDemo(ABC):
    """Abstract base class for running demo scripts with DQN and plotting results."""

    def __init__(self, env_id: str):
        """Initialize the base demo class with common settings."""
        self.env_wrapper = MultiAgentEnvironmentWrapper(env_id=env_id)
        self.num_agents = self.env_wrapper.num_agents
        self.episode_informations = [
            EpisodeInformation([], []) for _ in range(self.num_agents)
        ]
        self.plotter = Plotter() if settings.PLOTTING else None
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
            if self.plotter:
                self.plotter.update(self.episode_informations, show_result=True)
                plt.ioff()
                plt.show()

    @abstractmethod
    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Abstract method to handle running an episode. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_conv_layers(self, info) -> list[ConvLayer]:
        """Abstract method to get convolutional layers. Must be implemented by subclasses."""
        pass
