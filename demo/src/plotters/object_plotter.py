import matplotlib.pyplot as plt
import torch
from IPython import display

from demo.src.common import EpisodeInformation


class ObjectPlotter:
    """Handles plotting of training progress."""

    def __init__(self):
        self.fig, self.ax1 = plt.subplots()
        self.is_ipython = "inline" in plt.get_backend()

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
    ):
        """Update the plot with data from multiple episodes."""
        self.ax1.clear()

        if isinstance(episodes_information, EpisodeInformation):
            episodes_information = [episodes_information]

        self.ax1.clear()
        self._set_axis()

        for _, episode_info in enumerate(episodes_information):
            object_moved_distance_t = torch.tensor(
                episode_info.object_moved_distance, dtype=torch.float
            )
            self._plot_metrics(object_moved_distance_t)

        self.fig.tight_layout()
        plt.pause(0.001)

        if self.is_ipython:
            display.display(self.fig)

    def _set_axis(self):
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Samples")
        self.ax1.tick_params(axis="y")

    def _plot_metrics(
        self,
        object_moved_distance_t: torch.Tensor,
    ):
        """Plot metrics such as duration and rewards."""
        duration_means = self._moving_average(object_moved_distance_t)

        self.ax1.plot(object_moved_distance_t.numpy(), alpha=0.3)
        self.ax1.plot(duration_means.numpy())

    @staticmethod
    def _moving_average(tensor: torch.Tensor, window_size: int = 100):
        """Compute the moving average of a tensor."""
        len_averages = min(window_size, len(tensor))
        means = tensor.unfold(0, len_averages, 1).mean(1).view(-1)
        return torch.cat((torch.zeros(len_averages), means))
