import logging
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import network, settings
from demo.src.common import Batch
from demo.src.common.episode_information import EpisodeInformation
from demo.src.plotters import Plotter
from demo.src.wrappers import MultiAgentEnvironmentWrapper
from environments import settings as env_settings
from history import ModelHandler
from renderer import Renderer
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule


class BaseDemo(ABC):
    """Abstract base class for running demo scripts with DQN and plotting results."""

    def __init__(self, env_id: str):
        """Initialize the base demo class with common settings."""
        render_mode = "rgb_array" if settings.RENDER_Q_VALUES else "human"
        self.env_wrapper = MultiAgentEnvironmentWrapper(
            env_id=env_id, render_mode=render_mode
        )
        self.num_agents = self.env_wrapper.num_agents
        self.episode_informations = [
            EpisodeInformation([], []) for _ in range(self.num_agents)
        ]
        self.plotter = Plotter() if settings.PLOTTING else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ipython = "inline" in matplotlib.get_backend()

        self.model_handler = ModelHandler()
        self.extern_renderer = self._create_extern_renderer()

        self.q_value_renderer_agent_id = 0

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

    def render(self):
        """Render the environment."""
        if settings.RENDER_Q_VALUES:
            self._render_q_values()
        else:
            self.env_wrapper.render()

    def _train_batch(self, batch: Batch, agent_id: int):
        """Train the DQN with a batch of transitions."""
        self.dqns[agent_id].train(
            batch.states,
            batch.actions,
            batch.observations,
            batch.rewards,
            batch.terminated,
            batch.truncated,
        )

    @abstractmethod
    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Abstract method to handle running an episode. Must be implemented by subclasses."""
        pass

    def _render_q_values(self):
        if self.extern_renderer is None:
            raise ValueError("External renderer not initialized")
        rgb_array = self.env_wrapper.render()
        if rgb_array is None:
            raise ValueError("rgb array should not be None")
        states = self.env_wrapper.get_all_possible_states()
        q_values = self.dqns[self.q_value_renderer_agent_id].get_q_values_map(states)
        self.extern_renderer.render(background=rgb_array, q_values=q_values)

    def _create_extern_renderer(self) -> Renderer | None:
        env_height = env_settings.ENV_HEIGHT
        env_width = env_settings.ENV_WIDTH
        screen_width = env_settings.SCREEN_WIDTH
        screen_height = env_settings.SCREEN_HEIGHT
        return (
            Renderer(env_height, env_width, screen_width, screen_height)
            if settings.RENDER_Q_VALUES
            else None
        )

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
        dqn = DQNModule(
            observation_shape,
            n_actions,
            hidden_layers=network.HIDDEN_LAYERS,
            conv_layers=conv_layers,
        )
        self.dqns = [dqn] * self.num_agents

        if settings.PRETRAINED:
            for i, dqn in enumerate(self.dqns):
                self.model_handler.load(dqn, f"{settings.LOAD_MODEL_NAME}_agent{i}")

    def _save_models(self, iteration: int):
        """Save the DQN models for each agent."""
        model_name = settings.SAVE_MODEL_NAME + f"_{iteration}"
        for i, dqn in enumerate(self.dqns):
            self.model_handler.save(dqn, f"{model_name}_agent{i}")
