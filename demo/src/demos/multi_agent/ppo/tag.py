from itertools import count

import numpy as np
import torch

from demo import settings
from demo.src.common import Batch, Transition

from .base_demo import BaseDemo


class TagDemo(BaseDemo):
    """Class for running the Tag demo with DQN and plotting results."""

    def __init__(self):
        """Initialize the Tag demo."""
        super().__init__(env_id="TagEnv-v0")

    def _run_episode(self, i_episode: int, state: torch.Tensor, info: dict):
        """Handle the episode by interacting with the environment and training the DQN."""
        state, _ = self.env_wrapper.reset()

        seeker_batch = Batch(
            states=[],
            actions=[],
            observations=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )
        hider_batch = Batch(
            states=[],
            actions=[],
            observations=[],
            rewards=[],
            terminated=[],
            truncated=[],
        )
        agent_batches = [seeker_batch, hider_batch]
        total_rewards = np.zeros(self.num_agents)

        for t in count():
            done = False
            full_states, transitions, info = self._run_step(state, t)

            for agent, transition in enumerate(transitions):
                done = done or transition.terminated or transition.truncated

            if not done:
                full_state, agent_rewards, done = self.env_wrapper.concatenate_states(
                    full_states
                )
                state = self.env_wrapper.update_state(full_state[0].numpy())

                for agent, transition in enumerate(transitions):
                    transition.reward += agent_rewards[agent]

            for agent, transition in enumerate(transitions):
                agent_batches[agent].append(transition)

            if i_episode % settings.RENDER_EVERY == 0:
                self.render()

            for agent in range(len(transitions)):
                total_rewards[agent] += transitions[agent].reward.item()

            if done:
                object_moved_distance = info.get("object_moved_distance")
                if object_moved_distance is None:
                    raise ValueError(
                        "Object moved distance must be returned from the environment."
                    )
                self.episode_informations[1].object_moved_distance.append(
                    object_moved_distance
                )
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
    ) -> tuple[list[np.ndarray], list[Transition], dict]:
        """Run a step in the environment and train the DQN."""
        full_states = []
        transitions = []

        object_moved_distance = 0

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
            new_object_moved_distance = info.get("object_moved_distance")
            if new_object_moved_distance is not None:
                object_moved_distance += new_object_moved_distance

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
            transitions.append(transition)

        self.env_wrapper.set_active_agent(0)

        return (
            full_states,
            transitions,
            {"object_moved_distance": object_moved_distance},
        )


if __name__ == "__main__":
    demo = TagDemo()
    demo.run()
