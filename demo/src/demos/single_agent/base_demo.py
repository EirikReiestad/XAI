from abc import ABC, abstractmethod
import logging
from itertools import count

import matplotlib.pyplot as plt
import torch

from demo import network, settings
from demo.src.common import EpisodeInformation
from demo.src.plotters import Plotter
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule
from demo.src.wrappers.single_agent_environment_wrapper import (
    SingleAgentEnvironmentWrapper,
)


class BaseDemo(ABC):
    """Abstract base class for running demos with DQN and plotting results."""

    def __init__(self):
        """Initialize the base demo class with common settings."""
        self.episode_information = EpisodeInformation(durations=[], rewards=[])
        self.plotter = Plotter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ipython = "inline" in plt.get_backend()

        self.env_wrapper = self._create_environment_wrapper()

    def run(self):
        """Run the demo, interacting with the environment and training the DQN."""
        state, info = self.env_wrapper.reset()
        n_actions = self.env_wrapper.action_space.n
        conv_layers = self._create_conv_layers(info)

        self.dqn = DQNModule(state.shape, n_actions, conv_layers=conv_layers)

        plt.ion()

        try:
            for i_episode in range(settings.NUM_EPISODES):
                self._run_episode(i_episode, state, info)
        except Exception as e:
            logging.exception(e)
        finally:
            self.env_wrapper.close()
            logging.info("Complete")
            self.plotter.update(self.episode_information, show_result=True)
            plt.ioff()
            plt.show()

    @abstractmethod
    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        pass

    @abstractmethod
    def _create_environment_wrapper(self) -> SingleAgentEnvironmentWrapper:
        """Abstract method to create an environment wrapper. Must be implemented by subclasses."""
        pass

    def _create_conv_layers(self, info) -> list[ConvLayer]:
        """Create convolutional layers based on the state type."""
        state_type = info.get("state_type") if info else None
        if state_type in {"rgb", "full"}:
            return network.CONV_LAYERS
        return []
