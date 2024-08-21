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

    def update(
        self,
        episodes_information: list[EpisodeInformation] | EpisodeInformation,
        labels: list[str] | str = [],
        show_result=False,
    ):
        """Update the plot with data from multiple episodes."""
        self.ax1.clear()

        if isinstance(episodes_information, EpisodeInformation):
            episodes_information = [episodes_information]

        self.ax1.clear()
        self.ax2.clear()

        if len(labels) == 0:
            labels = [f"Agent {i}" for i in range(len(episodes_information))]

        colors = plt.cm.get_cmap("tab10", len(episodes_information))

        self._set_axis()

        for i, episode_info in enumerate(episodes_information):
            durations_t = torch.tensor(episode_info.durations, dtype=torch.float)
            rewards_t = torch.tensor(episode_info.rewards, dtype=torch.float)
            self._plot_metrics(durations_t, rewards_t, label=labels[i], color=colors(i))

        self.fig.tight_layout()
        plt.pause(0.001)

        if self.is_ipython:
            if not show_result:
                display.display(self.fig)
                display.clear_output(wait=True)
            else:
                display.display(self.fig)

    def _set_axis(self):
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Duration")
        self.ax1.tick_params(axis="y")

        self.ax2.set_ylabel("Rewards")
        self.ax2.yaxis.set_label_position("right")
        self.ax2.tick_params(axis="y")

    def _plot_metrics(
        self,
        durations_t: torch.Tensor,
        rewards_t: torch.Tensor,
        label: str,
        color,
    ):
        """Plot metrics such as duration and rewards."""
        duration_means = self._moving_average(durations_t)
        rewards_means = self._moving_average(rewards_t)

        self.ax1.plot(
            durations_t.numpy(), color="orange", alpha=0.3, label=f"{label} Duration"
        )
        self.ax1.plot(
            duration_means.numpy(),
            color="orange",
        )
        self.ax1.legend(loc="upper left")
        self.ax2.plot(
            rewards_t.numpy(),
            color=color,
            alpha=0.3,
            label=f"{label} Rewards",
        )
        self.ax2.plot(rewards_means.numpy(), color=color)
        self.ax2.legend(loc="upper right")

    @staticmethod
    def _moving_average(tensor: torch.Tensor, window_size: int = 100):
        """Compute the moving average of a tensor."""
        len_averages = min(window_size, len(tensor))
        means = tensor.unfold(0, len_averages, 1).mean(1).view(-1)
        return torch.cat((torch.zeros(len_averages), means))
