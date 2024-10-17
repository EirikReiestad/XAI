import gymnasium as gym
import numpy as np


class CAVWrapper:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_concepts(self) -> dict:
        return self.env.unwrapped.concepts

    def get_concept_names(self) -> list[str]:
        return self.env.unwrapped.concept_names

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        if concept not in self.get_concept_names():
            raise ValueError(f"Concept {concept} not found in environment.")
        state, label = self.env.unwrapped.get_concept(concept, samples)
        return state, label
