from .scopus_reader import ScopusReader
from .writer import Writer


class Handler:
    def __init__(self, writer_filename: str):
        self.scopus_reader = ScopusReader()
        self.writer = Writer(writer_filename)

    def run(self, filename: str):
        data = self.scopus_reader.read(filename)
        self.writer.write(data)
