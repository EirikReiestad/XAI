import pandas as pd


class Writer:
    def __init__(self, path):
        self.path = path

    def write(self, data: pd.DataFrame):
        data.to_csv(self.path, mode="a", header=False, index=False)
