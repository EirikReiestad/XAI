import os
from dataclasses import dataclass

import numpy as np


@dataclass
class Sample:
    id: str
    data: np.ndarray
    label: str

    def __str__(self) -> str:
        return f"Sample {self.id}: {self.data}, {self.label}"


class Data:
    samples: list[Sample]

    def __init__(
        self,
        filename: str | None = None,
        folder_path: str = "data_handler/data/",
    ) -> None:
        self.folder_path = folder_path
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Folder {folder_path} not found.")
        self.load_data_from_filename(filename)

    def load_data_from_filename(self, filename: str | None) -> None:
        if filename is not None:
            self.load_samples_from_filename(filename)

    def load_samples_from_filename(self, filename: str) -> None:
        file_path = os.path.join(self.folder_path, filename)
        self.samples = self._sample_from_file(file_path)

    def load_samples(self, samples: list[Sample]) -> None:
        self.samples = samples

    def _sample_from_file(self, path: str) -> list[Sample]:
        self._check_file_exists(path)
        with open(path, "r") as f:
            samples = []
            for line in f:
                id, *data, label = line.strip().split(",")
                data = np.array(
                    [
                        np.fromstring(row.strip("[]"), sep=" ")
                        for row in data[0].split(";")
                    ]
                )
                samples.append(Sample(id, np.array(data), label))
        return samples

    def _check_file_exists(self, path: str) -> None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} not found.")

    def save(self, filename: str) -> None:
        filepath = os.path.join(self.folder_path, filename)
        self.write_data_to_file(filepath)

    def write_data_to_file(self, path: str) -> None:
        with open(path, "w") as f:
            for sample in self.samples:
                if sample.data.ndim == 1:
                    raise ValueError("Data must be a matrix.")
                matrix_str = ";".join(
                    ["[" + " ".join(map(str, row)) + "]" for row in sample.data]
                )
                f.write(f"{sample.id},{matrix_str},{sample.label}\n")

    def get_data_lists(self) -> tuple[list[np.ndarray], list[str]]:
        states = []
        labels = []
        for sample in self.samples:
            states.append(sample.data)
            labels.append(sample.label)
        return states, labels

    def split(self, ratio: float) -> tuple["Data", "Data"]:
        n_samples = len(self.samples)
        n_samples1 = int(n_samples * ratio)
        samples1 = self.samples[:n_samples1]
        samples2 = self.samples[n_samples1:]
        data1 = Data()
        data1.load_samples(samples1)
        data2 = Data()
        data2.load_samples(samples2)
        return data1, data2

    def __str__(self) -> str:
        return str(self.samples)