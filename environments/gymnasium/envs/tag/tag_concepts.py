from dataclasses import dataclass
import numpy as np
from .tag_state import TagState


@dataclass
class Concept:
    name: str
    description: str


class TagConcepts:
    concepts = [
        Concept(
            "box-block",
            "A block that can be pushed around by the agent. It should be blocking thee path between the agents.",
        ),
        Concept(
            "random", "An arbitrary concept that does not have a specific meaning."
        ),
    ]

    def __init__(self, state: TagState, num_actions: int) -> None:
        self.state = state
        self.num_actions = num_actions

    @property
    def concept_names(self) -> list[str]:
        return [concept.name for concept in self.concepts]

    def get_concepts_dict(self) -> dict:
        return {concept.name: concept.description for concept in self.concepts}

    def get_concept(
        self, concept: str, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        if concept not in self.concept_names:
            raise ValueError(f"Concept {concept} not found in environment.")
        if concept == "box-block":
            return self._get_box_block_concept(samples)
        elif concept == "random":
            return self._get_random_concept(samples)
        else:
            raise NotImplementedError(f"Concept {concept} is not implemented yet.")

    def _get_box_block_concept(
        self, samples: int
    ) -> tuple[list[np.ndarray], list[str]]:
        return self._get_random_concept(samples)

    def _get_random_concept(self, samples: int) -> tuple[list[np.ndarray], list[str]]:
        self.state.random_seeker_position = True
        self.state.random_hider_position = True
        states = []
        labels = []
        for _ in range(samples):
            self.state.reset()
            states.append(self.state.full)
            labels.append(np.random.randint(self.num_actions))
        return states, labels
