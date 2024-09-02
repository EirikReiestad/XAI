from demo.src.demos.multi_agent.base_demo import BaseDemo
import torch
from demo import settings
from itertools import count
import numpy as np
from demo.src.common import Batch, Transition


class CoopDemo(BaseDemo):
    """Class for running the Coop demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Coop demo."""
        super().__init__(env_id="CoopEnv-v0")

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Handle the episode by interacting with the environment and training the DQN."""
        state, _ = self.env_wrapper.reset()
        total_rewards = np.zeros(self.num_agents)

        agent0_batch = Batch(
            states=[],
            actions=[],
            observations=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )
        agent1_batch = Batch(
            states=[],
            actions=[],
            observations=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )
        agent_batches = [agent0_batch, agent1_batch]

        for t in count():
            done = False
            full_states, transitions = self._run_step(state, t)
            for agent, transition in enumerate(transitions):
                total_rewards[agent] += transition.reward.item()
                done = done or transition.terminated or transition.truncated

            if not done:
                full_state, agent_rewards, done = self.env_wrapper.concatenate_states(
                    full_states
                )
                state = self.env_wrapper.update_state(full_state[0].numpy())

                for agent, transition in enumerate(transitions):
                    transition.reward += agent_rewards[agent]

            if i_episode % settings.RENDER_EVERY == 0:
                self.render()

            for agent, transition in enumerate(transitions):
                agent_batches[agent].append(transition)

            if done:
                for agent in range(self.num_agents):
                    self.episode_informations[agent].durations.append(t + 1)
                    self.episode_informations[agent].rewards.append(
                        total_rewards[agent]
                    )
                if self.plotter:
                    self.plotter.update(self.episode_informations)
                break

        for agent in range(self.num_agents):
            self._train_batch(agent_batches[agent], agent)

    def _run_step(
        self, state: torch.Tensor, step: int
    ) -> tuple[list[np.ndarray], list[Transition]]:
        """Run a step in the environment and train the DQN."""
        full_states = []
        agent_transitions = []

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

            transition = Transition(
                state=state,
                action=action,
                observation=observation,
                reward=torch.tensor([reward], dtype=torch.float32),
                terminated=terminated,
                truncated=truncated,
            )

            agent_transitions.append(transition)

        self.env_wrapper.set_active_agent(0)

        return full_states, agent_transitions


if __name__ == "__main__":
    demo = CoopDemo()
    demo.run()
