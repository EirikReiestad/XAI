import matplotlib.pyplot as plt
import torch
from IPython import display

from demo.src.common import EpisodeInformation


class Plotter:
    """Handles plotting of training progress."""

    def __init__(self):
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()
        self.is_ipython = "inline" in plt.get_backend()

    def update(self, episode_information: EpisodeInformation, show_result=False):
        """Update the plot with the latest data."""
        self.ax1.clear()
        self.ax2.clear()

        durations_t = torch.tensor(episode_information.durations, dtype=torch.float)
        rewards_t = torch.tensor(episode_information.rewards, dtype=torch.float)

        self._plot_metrics(durations_t, rewards_t)
        self.fig.tight_layout()
        plt.pause(0.001)

        if self.is_ipython:
            if not show_result:
                display.display(self.fig)
                display.clear_output(wait=True)
            else:
                display.display(self.fig)

    def _plot_metrics(self, durations_t: torch.Tensor, rewards_t: torch.Tensor):
        """Plot metrics such as duration and rewards."""
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Duration", color="tab:orange")
        self.ax1.tick_params(axis="y", labelcolor="tab:orange")

        self.ax2.set_ylabel("Rewards", color="tab:cyan")
        self.ax2.yaxis.set_label_position("right")
        self.ax2.tick_params(axis="y", labelcolor="tab:cyan")

        duration_means = self._moving_average(durations_t)
        rewards_means = self._moving_average(rewards_t)

        self.ax1.plot(duration_means.numpy(), color="tab:orange")
        self.ax2.plot(rewards_means.numpy(), color="tab:cyan")

    @staticmethod
    def _moving_average(tensor: torch.Tensor, window_size: int = 100):
        """Compute the moving average of a tensor."""
        len_averages = min(window_size, len(tensor))
        means = tensor.unfold(0, len_averages, 1).mean(1).view(-1)
        return torch.cat((torch.zeros(len_averages), means))
