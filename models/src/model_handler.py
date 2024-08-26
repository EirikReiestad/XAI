import os
from datetime import datetime

import torch
from matplotlib.figure import Figure

from demo import DemoType, settings


class ModelHandler:
    def __init__(self):
        dt = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder_name = self._get_folder_name()
        self.save_folder = self._create_folder(folder_name, dt)

    def load(self, model: torch.nn.Module, name: str):
        model.load_state_dict(torch.load(os.path.join(self.save_folder, name)))
        return model

    def save(self, model: torch.nn.Module, name: str):
        torch.save(model.state_dict(), os.path.join(self.save_folder, name))

    def save_plot(self, plot: Figure, filename: str = "plot.png"):
        plot.savefig(os.path.join(self.save_folder, filename))

    def _get_folder_name(self):
        if settings.DEMO == DemoType.MAZE:
            return "maze"
        elif settings.DEMO == DemoType.COOP:
            return "coop"
        else:
            raise ValueError("Invalid demo type")

    def _create_folder(self, foldername: str, timestamp: str):
        base_path = os.path.join("models", "models", foldername)
        if not os.path.exists(base_path):
            raise OSError(f"Folder {base_path} does not exist")

        save_folder = os.path.join(base_path, timestamp)
        os.makedirs(save_folder, exist_ok=True)

        return save_folder
