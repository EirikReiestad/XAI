import numpy as np
from matplotlib.figure import Figure

from demo.src.common import EpisodeInformation

from .agent_plotter import AgentPlotter
from .object_plotter import ObjectPlotter
from demo import settings


class Plotter:
    """Handles plotting of training progress."""

    def __init__(self):
        if settings.PLOT_AGENT_REWARD:
            self.agent_plotter = AgentPlotter()
        if settings.PLOT_OBJECT_MOVEMENT:
            self.object_plotter = ObjectPlotter()

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
        colors: list[str] = ["blue", "lightgreen"],
        labels: list[str] | str = [],
        show_result=False,
    ):
        """Update the plot with data from multiple episodes."""
        if settings.PLOT_AGENT_REWARD:
            self.agent_plotter.update(
                episodes_information,
                colors=colors,
                labels=labels,
                show_result=show_result,
            )
        if settings.PLOT_OBJECT_MOVEMENT:
            self.object_plotter.update(episodes_information)

    def plot_q_values(self, q_values: np.ndarray):
        """Plot the Q-values of the DQN."""
        self.agent_plotter.plot_q_values(q_values)

    @property
    def figs(self) -> list[Figure]:
        figs = []
        if settings.PLOT_AGENT_REWARD:
            figs.append(self.agent_plotter.fig)
        if settings.PLOT_OBJECT_MOVEMENT:
            figs.append(self.object_plotter.fig)
        return figs
