import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch
import logging
from itertools import count
from IPython import display
import numpy as np

from rl.src.dqn.dqn_module import DQNModule
from environments.gymnasium.envs.maze.utils import preprocess_state
from demo.src.common import EpisodeInformation
from demo import settings

# Set up matplotlib
is_ipython = "inline" in matplotlib.get_backend()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Demo:
    def __init__(self) -> None:
        self.episode_information = EpisodeInformation(durations=[], rewards=[])
        self.fig, self.ax1 = plt.subplots()
        self.ax2 = self.ax1.twinx()

    def main(self):
        env = gym.make("CartPole-v1", render_mode="human")

        state, _ = env.reset()
        state = preprocess_state(state)

        dqn = DQNModule(state.shape, env.action_space.n, seed=4)

        plt.ion()

        num_episodes = settings.NUM_EPISODES

        try:
            for i_episode in range(num_episodes):
                state, _ = env.reset()
                state = preprocess_state(state)
                # state = torch.tensor(state, dtype=torch.float32, device=device)

                total_reward = 0

                for t in count():
                    if i_episode % settings.RENDER_EVERY == 0:
                        env.render()

                    action = dqn.select_action(state)
                    observation, reward, terminated, truncated, _ = env.step(
                        action.item()
                    )

                    total_reward += float(reward)

                    observation = np.array(observation)
                    observation = preprocess_state(observation)

                    # observation = torch.tensor(
                    #     observation, dtype=torch.float32, device=device
                    # )

                    done, state = dqn.train(
                        state, action, observation, float(reward), terminated, truncated
                    )

                    if done:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(total_reward)
                        self.plot()
                        break
        except Exeption as e:
            logging.error(e)
        finally:
            env.close()

            logging.info("Complete")
            self.plot(show_result=True)
            plt.ioff()
        plt.show()

    def plot(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(
            self.episode_information.durations, dtype=torch.float
        )
        if show_result:
            plt.title("Result")
        else:
            plt.clf()
            plt.title("Training...")
        plt.xlabel("Episode")
        plt.ylabel("Duration")
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())


if __name__ == "__main__":
    demo = Demo()
    demo.main()
