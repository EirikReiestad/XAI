import matplotlib
import matplotlib.pyplot as plt
import torch
import logging
from itertools import count
import numpy as np
import gymnasium as gym
from IPython import display

from rl.src.dqn.dqn_module import DQNModule
from environments.gymnasium.envs.maze.utils.preprocess_state import preprocess_state
from demo.src.common import EpisodeInformation

from demo.src import settings

gym.register(
    id="Maze-v0",
    entry_point="environments.gymnasium.envs.maze.maze:MazeEnv",
)

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Demo:
    def __init__(self):
        self.episode_information: EpisodeInformation = EpisodeInformation(
            durations=[], rewards=[]
        )

        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

    def main(self):
        env = gym.make("Maze-v0", render_mode="rgb_array")

        _, _ = env.reset()
        state = env.render()

        state = np.array(state)
        state = preprocess_state(state)

        model_path = "maze_dqn.pth"
        n_actions = env.action_space.n
        dqn = DQNModule(state.shape, n_actions, path=model_path, seed=4)

        plt.ion()

        num_episodes = settings.NUM_EPISODES
        render_every = settings.RENDER_EVERY
        try:
            for i_episode in range(num_episodes):
                _, _ = env.reset()
                state = env.render()
                state = np.array(state)
                state = preprocess_state(state)

                total_reward = 0

                for t in count():
                    if i_episode % render_every == 0:
                        env.render(render_mode="human")

                    action = dqn.select_action(state)
                    _, reward, terminated, truncated, _ = env.step(action.item())

                    total_reward += float(reward)

                    observation = env.render()
                    observation = np.array(observation)
                    observation = preprocess_state(observation)

                    done, state = dqn.train(
                        state, action, observation, float(reward), terminated, truncated
                    )

                    if done:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(total_reward)
                        self.plot()
                        break

        except Exception as e:
            logging.exception(e)
        finally:
            env.close()

            logging.info("Complete")
            self.plot(show_result=True)
            plt.ioff()
        plt.show()

    def plot(self, show_result=False):
        self.ax1.clear()
        self.ax2.clear()

        durations_t = torch.tensor(
            self.episode_information.durations, dtype=torch.float
        )
        self.ax1.set_xlabel("Episode")
        self.ax1.set_ylabel("Duration", color="tab:orange")
        self.ax1.tick_params(axis="y", labelcolor="tab:blue")

        plt.title("Training...")

        self.ax2.set_ylabel("Rewards", color="tab:cyan")
        self.ax2.yaxis.set_label_position("right")
        rewards_t = torch.tensor(self.episode_information.rewards, dtype=torch.float)
        self.ax2.tick_params(axis="y", labelcolor="tab:cyan")

        # Take 100 episode averages and plot them too
        len_averages = min(10, len(durations_t))

        duration_means = durations_t.unfold(0, len_averages, 1).mean(1).view(-1)
        duration_means = torch.cat((torch.zeros(len_averages), duration_means))
        self.ax1.plot(duration_means.numpy(), color="tab:cyan")

        rewards_means = rewards_t.unfold(0, len_averages, 1).mean(1).view(-1)
        rewards_means = torch.cat((torch.zeros(len_averages), rewards_means))
        self.ax2.plot(rewards_means.numpy(), color="tab:orange")

        self.fig.tight_layout()  # To ensure the right y-label is not slightly clipped
        plt.pause(0.001)  # pause a bit so that plots are updated

        if is_ipython:
            if not show_result:
                display.display(self.fig)
                display.clear_output(wait=True)
            else:
                display.display(self.fig)


if __name__ == "__main__":
    demo = Demo()
    demo.main()
