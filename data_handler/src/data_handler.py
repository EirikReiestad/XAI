from .utils.data import Data, Sample
from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper
import numpy as np


class DataHandler:
    def __init__(self) -> None:
        self.data = Data()

    def load_data_from_path(self, filename: str) -> None:
        self.data.load_samples_from_filename(filename)

    def load_samples(self, samples: list[Sample]) -> None:
        self.data.load_samples(samples)

    def save(self, filename: str) -> None:
        self.data.save(filename)

    def _get_samples(self, states: list[np.ndarray], labels: list[str]) -> list[Sample]:
        samples: list[Sample] = []
        for i, (state, label) in enumerate(zip(states, labels)):
            samples.append(Sample(f"id_{i}", state, label))
        return samples

    def generate_data(self, env: CAVWrapper, concept: str, n_samples: int) -> None:
        states, labels = env.get_concept(concept, n_samples)
        samples = self._get_samples(states, labels)
        self.load_samples(samples)

    def get_data_lists(self, sample_ratio) -> tuple[list[np.ndarray], list[str]]:
        return self.data.get_data_lists(sample_ratio=sample_ratio)

    def split(
        self, ratio: float, bootstrapped: float = 1.0
    ) -> tuple["DataHandler", "DataHandler"]:
        data1, data2 = self.data.split(ratio, bootstrapped)
        dh1 = DataHandler()
        dh1.load_samples(data1.samples)
        dh2 = DataHandler()
        dh2.load_samples(data2.samples)
        return dh1, dh2

    def show_random_sample(self, n_samples: int) -> None:
        self.data.show_random_sample(n_samples)

    def __str__(self) -> str:
        return str(self.data)
