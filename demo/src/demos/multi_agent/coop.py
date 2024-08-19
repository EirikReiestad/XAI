import logging
from itertools import count
import gymnasium as gym
import matplotlib
import matplotlib.pyplot as plt
import torch

from demo import settings
from demo import network
from rl.src.common import ConvLayer
from rl.src.dqn.dqn_module import DQNModule
from demo.src.plotters import Plotter
from demo.src.wrappers import MultiAgentEnvironmentWrapper

# Register Gym environment
gym.register(
    id="Coop-v0",
    entry_point="environments.gymnasium.envs.coop.coop:CoopEnv",
)


class Demo:
    """Class for running the Maze demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Demo class with settings and plotter."""
        self.episode_information = EpisodeInformation(durations=[], rewards=[])
        self.plotter = Plotter()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_ipython = "inline" in matplotlib.get_backend()

        self.num_agents = 1

    def run(self):
        """Run the demo, interacting with the environment and training the DQN."""
        env_wrapper = MultiAgentEnvironmentWrapper(env_id="Coop-v0")
        state, info = env_wrapper.reset()
        n_actions = env_wrapper.action_space.n
        conv_layers = self._get_conv_layers(info)

        dqn = DQNModule(state.shape, n_actions, conv_layers=conv_layers)

        dqns = [dqn] * self.num_agents

        plt.ion()

        try:
            for i_episode in range(settings.NUM_EPISODES):
                state, _ = env_wrapper.reset()
                total_reward = 0

                for t in count():
                    if i_episode % settings.RENDER_EVERY == 0:
                        env_wrapper.render()

                    actions = [dqn.select_action(state) for dqn in dqns]

                    observations, rewards, terminated, truncated = env_wrapper.step(
                        actions.item()
                    )

                    reward = float(rewards)

                    total_reward += reward

                    dones, new_states = zip(
                        *[
                            dqn.train(
                                state,
                                action,
                                observation,
                                reward,
                                terminated,
                                truncated,
                            )
                            for dqn, action, observation, terminated, truncated in zip(
                                dqns, actions, observations, terminated, truncated
                            )
                        ]
                    )

                    state = self._concatenate_states(new_states)

                    done = any(dones)

                    if done:
                        self.episode_information.durations.append(t + 1)
                        self.episode_information.rewards.append(total_reward)
                        self.plotter.update(self.episode_information)
                        break

        except Exception as e:
            logging.exception(e)
        finally:
            env_wrapper.close()
            logging.info("Complete")
            self.plotter.update(self.episode_information, show_result=True)
            plt.ioff()
            plt.show()

    def _concatenate_states(self, states):
        """Concatenate states from multiple agents into a single state."""
        return torch.cat(states, dim=1)

    def _get_conv_layers(self, info) -> list[ConvLayer]:
        """Create convolutional layers based on the state type."""
        state_type = info.get("state_type") if info else None
        if state_type in {"rgb", "full"}:
            network.CONV_LAYERS
        return []


if __name__ == "__main__":
    demo = Demo()
    demo.run()
