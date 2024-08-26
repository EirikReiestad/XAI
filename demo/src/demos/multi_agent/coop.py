from demo.src.demos.multi_agent.base_demo import BaseDemo
import torch
from demo import settings
from itertools import count
import numpy as np


class CoopDemo(BaseDemo):
    """Class for running the Coop demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Coop demo."""
        super().__init__(env_id="CoopEnv-v0")

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Handle the episode by interacting with the environment and training the DQN."""
        state, _ = self.env_wrapper.reset()
        total_rewards = np.zeros(self.num_agents)
        dones = [False] * self.num_agents

        for t in count():
            _, rewards, dones, full_states = self._run_step(state, t)
            total_rewards += rewards

            done = any(dones)
            if not done:
                full_state, reward, done = self.env_wrapper.concatenate_states(
                    full_states
                )
                state = self.env_wrapper.update_state(full_state[0].numpy())

                for agent in range(self.num_agents):
                    if reward == 0:
                        break
                    total_rewards[agent] += reward

            if i_episode % settings.RENDER_EVERY == 0:
                self.env_wrapper.render()

            if done:
                for agent in range(self.num_agents):
                    self.episode_informations[agent].durations.append(t + 1)
                    self.episode_informations[agent].rewards.append(
                        total_rewards[agent]
                    )
                if self.plotter:
                    self.plotter.update(self.episode_informations)
                break

    def _run_step(
        self, state: torch.Tensor, step: int
    ) -> tuple[list[torch.Tensor], list[float], list[bool], list[np.ndarray]]:
        """Run a step in the environment and train the DQN."""
        total_reward = [0.0] * self.num_agents
        new_states = [torch.empty(0)] * self.num_agents
        full_states = []
        dones = [False] * self.num_agents

        for agent in range(self.num_agents):
            if agent == 1:
                if step % settings.SLOWING_FACTOR != 0:
                    continue

            action = self.dqns[agent].select_action(state)

            self.env_wrapper.set_active_agent(agent)
            observation, reward, terminated, truncated, info = self.env_wrapper.step(
                action.item()
            )

            full_state = info.get("full_state")
            if full_state is None:
                raise ValueError("Full state must be returned from the environment.")

            full_states.append(full_state)

            reward = float(reward)
            total_reward[agent] += reward

            done, new_state = self.dqns[agent].train(
                state,
                action,
                observation,
                reward,
                terminated,
                truncated,
            )

            if new_state is not None:
                new_states[agent] = new_state
            dones[agent] = done

        if any(new_state is None for new_state in new_states):
            raise ValueError("New state must be returned from the DQN.")

        return new_states, total_reward, dones, full_states


if __name__ == "__main__":
    demo = CoopDemo()
    demo.run()
