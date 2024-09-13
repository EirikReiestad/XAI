import numpy as np
from matplotlib.figure import Figure

from demo.src.common import EpisodeInformation

from .agent_plotter import AgentPlotter
from .object_plotter import ObjectPlotter


class Plotter:
    """Handles plotting of training progress."""

    def __init__(self, plot_agent: bool = False, plot_object: bool = False):
        self.plot_agent = plot_agent
        self.plot_object = plot_object
        if plot_agent:
            self.agent_plotter = AgentPlotter()
        if plot_object:
            self.object_plotter = ObjectPlotter()

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
        colors: list[str] = ["blue", "lightgreen"],
        labels: list[str] | str = [],
        show_result=False,
    ):
        """Update the plot with data from multiple episodes."""
        if self.plot_agent:
            self.agent_plotter.update(
                episodes_information,
                colors=colors,
                labels=labels,
                show_result=show_result,
            )
        if self.plot_object:
            self.object_plotter.update(episodes_information)

    def plot_q_values(self, q_values: np.ndarray):
        """Plot the Q-values of the DQN."""
        self.agent_plotter.plot_q_values(q_values)

    @property
    def figs(self) -> list[Figure]:
        figs = []
        if self.plot_agent:
            figs.append(self.agent_plotter.fig)
        if self.plot_object:
            figs.append(self.object_plotter.fig)
        return figs
