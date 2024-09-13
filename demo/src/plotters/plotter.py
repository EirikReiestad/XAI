import numpy as np
from matplotlib.figure import Figure

from demo.src.common import EpisodeInformation

from .agent_plotter import AgentPlotter
from .object_plotter import ObjectPlotter


class Plotter:
    """Handles plotting of training progress."""

    def __init__(
        self, plot_agent_reward: bool = False, plot_object_movement: bool = False
    ):
        if plot_agent_reward:
            self.agent_plotter = AgentPlotter()
        if plot_object_movement:
            self.object_plotter = ObjectPlotter()

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
        colors: list[str] = ["blue", "lightgreen"],
        labels: list[str] | str = [],
        show_result=False,
    ):
        """Update the plot with data from multiple episodes."""
        if self.agent_plotter:
            self.agent_plotter.update(
                episodes_information,
                colors=colors,
                labels=labels,
                show_result=show_result,
            )
        if self.object_plotter:
            self.object_plotter.update(episodes_information)

    def plot_q_values(self, q_values: np.ndarray):
        """Plot the Q-values of the DQN."""
        self.agent_plotter.plot_q_values(q_values)

    @property
    def figs(self) -> list[Figure]:
        figs = []
        if self.agent_plotter:
            figs.append(self.agent_plotter.fig)
        if self.object_plotter:
            figs.append(self.object_plotter.fig)
        return figs
