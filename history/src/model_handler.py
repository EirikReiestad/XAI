import os
from datetime import datetime

from matplotlib.figure import Figure

from demo import DemoType, settings
from rl.src.dqn.dqn_module import DQNModule


class ModelHandler:
    def __init__(self):
        self.dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.folder_name = self._get_folder_name()

    def load(self, model: DQNModule, name: str):
        if not hasattr(self, "load_folder"):
            self.load_folder = self._load_folder(self.folder_name)
        path = os.path.join(self.load_folder, name)
        model.load(path)

    def save(self, model: DQNModule, name: str):
        if not hasattr(self, "save_folder"):
            self.save_folder = self._save_folder(self.folder_name, self.dt)
        path = os.path.join(self.save_folder, name)
        model.save(path)

    def save_plot(self, plot: Figure, filename: str = "plot.png"):
        plot.savefig(os.path.join(self.save_folder, filename))

    def _get_folder_name(self):
        if settings.DEMO == DemoType.MAZE:
            return "maze"
        elif settings.DEMO == DemoType.COOP:
            return "coop"
        elif settings.DEMO == DemoType.TAG:
            return "tag"
        else:
            raise ValueError("Invalid demo type")

    def _save_folder(self, foldername: str, timestamp: str):
        base_path = os.path.join("history", "models", foldername)
        if not os.path.exists(base_path):
            raise OSError(f"Folder {base_path} does not exist")

        save_folder = os.path.join(base_path, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        return save_folder

    def _load_folder(self, foldername: str):
        base_path = os.path.join("history", "models", foldername)
        if not os.path.exists(base_path):
            raise OSError(f"Folder {base_path} does not exist")
        return base_path