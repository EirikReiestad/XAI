import logging
import os
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from demo import settings
from demo.src.common.episode_information import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import MultiAgentEnvironmentWrapper
from models import ModelHandler
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

        self.model_handler = ModelHandler()

    def run(self):
        """Run the demo, interacting with the environment and training the DQN."""
        state, info = self.env_wrapper.reset()
        n_actions = self.env_wrapper.action_space.n
        conv_layers = self._get_conv_layers(info)

        self._init_models(state.shape, n_actions, conv_layers)

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

    def _init_models(
        self, observation_shape: tuple, n_actions: int, conv_layers: list[ConvLayer]
    ):
        """Initialize the DQN models for each agent."""
        dqn = DQNModule(observation_shape, n_actions, conv_layers=conv_layers)
        self.dqns = [dqn] * self.num_agents

        if settings.PRETRAINED:
            model_name_agent0 = os.path.join(settings.MODEL_NAME, "_agent0")
            model_name_agent1 = os.path.join(settings.MODEL_NAME, "_agent1")

            self.model_handler.load_models(
                self.dqns, [model_name_agent0, model_name_agent1]
            )

    def save_models(self):
        """Save the DQN models for each agent."""
        model_name = settings.MODEL_NAME
        for i, dqn in enumerate(self.dqns):
            self.model_handler.save(dqn, f"{model_name}_agent{i}")

    @abstractmethod
    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Abstract method to handle running an episode. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _get_conv_layers(self, info) -> list[ConvLayer]:
        """Abstract method to get convolutional layers. Must be implemented by subclasses."""
        pass
