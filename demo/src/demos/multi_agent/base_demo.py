import logging
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import network, settings
from demo.src.common.episode_information import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import MultiAgentEnvironmentWrapper
from history import ModelHandler
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

        self._load_models(state.shape, n_actions, conv_layers)

        plt.ion()

        try:
            for i_episode in range(settings.NUM_EPISODES):
                self._run_episode(i_episode, state, info)
                if (
                    settings.SAVE_MODEL
                    and i_episode % settings.SAVE_EVERY == 0
                    and i_episode > 0
                ):
                    self._save_models(i_episode)
                    self._save_plot()
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

    def _get_conv_layers(self, info) -> list[ConvLayer]:
        """Create convolutional layers based on the state type."""
        state_type = info.get("state_type") if info else None
        if state_type in {"rgb", "full"}:
            return network.CONV_LAYERS
        return []

    def _save_plot(self):
        """Save the plot of the episode information."""
        if self.plotter is not None:
            self.model_handler.save_plot(self.plotter.fig, "plot")
        else:
            logging.warning("Plotter is not initialized. Cannot save plot.")

    def _load_models(
        self, observation_shape: tuple, n_actions: int, conv_layers: list[ConvLayer]
    ):
        """Initialize the DQN models for each agent."""
        dqn = DQNModule(observation_shape, n_actions, conv_layers=conv_layers)
        self.dqns = [dqn] * self.num_agents

        if settings.PRETRAINED:
            for i, dqn in enumerate(self.dqns):
                self.model_handler.load(dqn, f"{settings.LOAD_MODEL_NAME}_agent{i}")

    def _save_models(self, iteration: int):
        """Save the DQN models for each agent."""
        model_name = settings.SAVE_MODEL_NAME + f"_{iteration}"
        for i, dqn in enumerate(self.dqns):
            self.model_handler.save(dqn, f"{model_name}_agent{i}")
