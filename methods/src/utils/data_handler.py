from methods.src.utils.data import Data, Sample


class DataHandler:
    def __init__(self) -> None:
        self.data = Data()

    def load_data_from_path(
        self, positive_samples_path: str, negative_samples_path: str
    ) -> None:
        self.data = Data(positive_samples_path, negative_samples_path)

    def load_positive_samples_from_path(self, path: str) -> None:
        self.data.load_positive_samples_from_path(path)

    def load_negative_samples_from_path(self, path: str) -> None:
        self.data.load_negative_samples_from_path(path)

    def load_positive_samples(self, samples: list[Sample]) -> None:
        self.data.load_positive_samples(samples)

    def load_negative_samples(self, samples: list[Sample]) -> None:
        self.data.load_negative_samples(samples)

    def write(self, filename: str) -> None:
        self.data.write(filename)


if __name__ == "__main__":
    positive_path = "data/positive_samples.csv"
    negative_path = "data/negative_samples.csv"
    data_handler = DataHandler()
    data_handler.load_data_from_path(positive_path, negative_path)
