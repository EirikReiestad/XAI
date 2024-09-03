import numpy as np

from demo.src.common import EpisodeInformation

from .agent_plotter import AgentPlotter
from .object_plotter import ObjectPlotter


class Plotter:
    """Handles plotting of training progress."""

    def __init__(self):
        self.agent_plotter = AgentPlotter()
        self.object_plotter = ObjectPlotter()

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
        colors: list[str] = ["blue", "lightgreen"],
        labels: list[str] | str = [],
        show_result=False,
    ):
        """Update the plot with data from multiple episodes."""
        self.agent_plotter.update(
            episodes_information, colors=colors, labels=labels, show_result=show_result
        )
        self.object_plotter.update(episodes_information)

    def plot_q_values(self, q_values: np.ndarray):
        """Plot the Q-values of the DQN."""
        self.agent_plotter.plot_q_values(q_values)
