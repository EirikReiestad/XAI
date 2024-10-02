import random
from itertools import count

import gymnasium as gym
import numpy as np
import torch

import rl

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sample_states(
    env: gym.Env,
    model: rl.SingleAgentBase | rl.MultiAgentBase,
    num_states: int = 1000,
    test: float = 0.2,
    sample_prob: float = 0.4,
):
    states = generate_states(env, model, num_states, sample_prob)
    background_states = states[: int(num_states * (1 - test))]
    test_states = states[int(num_states * (1 - test)) :]
    return background_states, test_states


def generate_states(
    env: gym.Env,
    model: rl.SingleAgentBase | rl.MultiAgentBase,
    num_states: int,
    sample_prob: float,
) -> np.ndarray:
    if isinstance(model, rl.SingleAgentBase):
        return generate_single_agent_states(env, model, num_states, sample_prob)
    elif isinstance(model, rl.MultiAgentBase):
        return generate_multi_agent_states(env, model, num_states, sample_prob)


def generate_single_agent_states(
    env: gym.Env, model: rl.SingleAgentBase, num_states: int, sample_prob: float
) -> np.ndarray:
    states: list[np.ndarray] = []

    state, _ = env.reset()
    for _ in count():
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        for _ in count():
            action = model.predict_action(state)
            observation, _, terminated, truncated, _ = env.step(action.item())
            state = torch.tensor(
                observation, device=device, dtype=torch.float32
            ).unsqueeze(0)

            if len(states) >= num_states:
                return np.vstack(states)

            if random.random() < sample_prob:
                numpy_state = state.cpu().numpy()
                states.append(numpy_state)

            if terminated or truncated:
                break

    return np.vstack(states)


def generate_multi_agent_states(
    env: gym.Env, model: rl.MultiAgentBase, num_states: int, sample_prob: float
) -> np.ndarray:
    states: list[np.ndarray] = []

    for _ in count():
        state, _ = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0)

        for _ in count():
            predicted_actions = model.predict_actions(state)
            actions = [action.item() for action in predicted_actions]
            (_, observation, terminated, _, _, _, truncated, _) = env.get_wrapper_attr(
                "step_multiple"
            )(actions)

            state = torch.tensor(
                observation, device=device, dtype=torch.float32
            ).unsqueeze(0)

            if len(states) >= num_states:
                return np.vstack(states)

            if random.random() < sample_prob:
                numpy_state = state.cpu().numpy()
                states.append(numpy_state)

            if terminated or truncated:
                break

    return np.vstack(states)
