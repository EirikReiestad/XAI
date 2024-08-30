from .literature_review_dataclass import LiteratureReviewDataclass
from .reader import Reader
import pandas as pd


class ScopusReader(Reader):
    def read_data(self, data: pd.DataFrame) -> LiteratureReviewDataclass:
        authors = data["Authors"].values[0]
        doi = data["DOI"].values[0]
        title = data["Title"].values[0]
        year = data["Year"].values[0]
        source_title = data["Source title"].values[0]
        cited_by = data["Cited by"].values[0]
        link = data["Link"].values[0]

        return LiteratureReviewDataclass(
            authors=authors,
            doi=doi,
            title=title,
            year=year,
            source_title=source_title,
            cited_by=cited_by,
            link=link,
        )
