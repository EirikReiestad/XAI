import gymnasium as gym
import torch


class TCAVWrapper:
    def __init__(self, env: gym.Env) -> None:
        self.env = env

    def get_concept_inputs(
        self, concept: str, samples: int
    ) -> tuple[list[torch.Tensor], list[int]]:
        """
        Get inputs for a concept.
            Args:
                concept: The concept to get inputs for.
                samples: The number of samples to get.
            Returns:
                A dictionary containing the inputs for the concept and the labels.
        """
        return [torch.rand(1, 1, 84, 84)], [0]
