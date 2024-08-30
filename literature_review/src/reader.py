from abc import ABC, abstractmethod
import pandas as pd
from .literature_review_dataclass import LiteratureReviewDataclass


class Reader(ABC):
    def read(self, filename: str) -> pd.DataFrame:
        csv = pd.read_csv(filename)
        scopus_data = self.read_data(csv)
        data_dict = scopus_data.dict
        print(data_dict)
        df = pd.DataFrame(data_dict)
        return df

    @abstractmethod
    def read_data(self, data: pd.DataFrame) -> LiteratureReviewDataclass:
        pass
