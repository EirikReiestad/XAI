import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    id: str
    data: np.ndarray
    label: str


class Data:
    positive_samples: list[Sample]
    negative_samples: list[Sample]

    def __init__(
        self,
        positive_sample_path: str | None = None,
        negative_sample_path: str | None = None,
        folder_path: str = "data/",
    ) -> None:
        self.folder_path = folder_path

        if positive_sample_path is not None:
            self.load_positive_samples_from_path(positive_sample_path)
        if negative_sample_path is not None:
            self.load_negative_samples_from_path(negative_sample_path)

    def load_positive_samples_from_path(self, path: str) -> None:
        self.positive_samples = self._sample_from_file(path)

    def load_negative_samples_from_path(self, path: str) -> None:
        self.negative_samples = self._sample_from_file(path)

    def load_positive_samples(self, samples: list[Sample]) -> None:
        self.positive_samples = samples

    def load_negative_samples(self, samples: list[Sample]) -> None:
        self.negative_samples = samples

    def _sample_from_file(self, path: str) -> list[Sample]:
        self._check_file_exists(path)
        with open(path, "r") as f:
            samples = []
            for line in f:
                id, *data, label = line.strip().split(",")
                samples.append(Sample(id, np.array(data), label))
        return samples

    def _check_file_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

    def write(self, filename: str) -> None:
        positive_file_path = os.path.join(
            self.folder_path, "positive_samples", filename
        )
        negative_file_path = os.path.join(
            self.folder_path, "negative_samples", filename
        )
        self.write_data_to_file(positive_file_path)
        self.write_data_to_file(negative_file_path)

    def write_data_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            for sample in self.positive_samples:
                f.write(
                    f"{sample.id},{','.join(map(str, sample.data))},{sample.label}\n"
                )
            for sample in self.negative_samples:
                f.write(
                    f"{sample.id},{','.join(map(str, sample.data))},{sample.label}\n"
                )


if __name__ == "__main__":
    data = Data("data/positive_samples.csv", "data/negative_samples.csv")
    print(data.positive_samples)
    print(data.negative_samples)
