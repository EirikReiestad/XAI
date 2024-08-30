from dataclasses import dataclass


@dataclass
class LiteratureReviewDataclass:
    authors: str
    doi: str
    title: str
    year: int
    source_title: str
    cited_by: int
    link: str

    @property
    def dict(self):
        return {
            "authors": self.authors,
            "doi": self.doi,
            "title": self.title,
            "year": self.year,
            "source_title": self.source_title,
            "cited_by": self.cited_by,
            "link": self.link,
        }
