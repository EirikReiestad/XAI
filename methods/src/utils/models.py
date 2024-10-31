import json
import logging
import os
import re

import torch

from rl.src.dqn.policies import DQNPolicy


class Models:
    def __init__(
        self,
        model: DQNPolicy,
        folder_suffix: str = "",
    ):
        self._extract_models(folder_suffix)
        self._model = model
        self.reset()
        self.next()

    def reset(self):
        self._model_idx = 0
        self._load_model()

    def _extract_models(self, folder_suffix: str):
        model_folder = os.path.join("models", "latest" + folder_suffix)
        metadata_folder = os.path.join("models", "metadata" + folder_suffix)

        assert os.path.exists(
            model_folder
        ), f"Model folder {model_folder} does not exist."
        assert os.path.exists(
            metadata_folder
        ), f"Metadata folder {metadata_folder} does not exist."

        self.pt_files = []
        self.metadata = []

        for root, _, files in os.walk(model_folder):
            for file in files:
                if file.endswith(".pt"):
                    pt_file_path = os.path.join(root, file)
                    self.pt_files.append(pt_file_path)

        for root, _, files in os.walk(metadata_folder):
            for file in files:
                if file.endswith(".json"):
                    json_file_path = os.path.join(root, file)
                    with open(json_file_path, "r") as f:
                        json_data = json.load(f)
                        self.metadata.append((json_file_path, json_data))

        combined = []

        def extract_model_number(path):
            match = re.search(r"model_(\d+)", path)
            return int(match.group(1)) if match else float("inf")

        for pt_file in self.pt_files:
            model_number = extract_model_number(pt_file)
            metadata_entry = next(
                (
                    m
                    for m in self.metadata
                    if extract_model_number(m[0]) == model_number
                ),
                None,
            )
            combined.append((pt_file, metadata_entry))

        combined.sort(key=lambda x: extract_model_number(x[0]))

        self.pt_files = [item[0] for item in combined]
        self.metadata = [item[1][1] for item in combined if item[1] is not None]

        assert len(self.pt_files) == len(self.metadata)

    def next(self):
        self._load_model()
        self._model_idx += 1

    def _load_model(self):
        self._model.policy_net.load_state_dict(
            torch.load(self.pt_files[self._model_idx], weights_only=True)
        )
        self._model.target_net.load_state_dict(self._model.policy_net.state_dict())
        self._model.policy_net.eval()
        self._model.target_net.eval()

    def has_next(self):
        return self._model_idx < len(self.pt_files) - 1

    @property
    def current_model_steps(self):
        if "steps_done" not in self.metadata[self._model_idx]:
            logging.error("No steps_done in metadata.")
            logging.error(self.metadata[self._model_idx])
            return 0
        return self.metadata[self._model_idx]["steps_done"]

    @property
    def policy_net(self):
        return self._model.policy_net

    @property
    def target_net(self):
        return self._model.target_net

    @property
    def model(self):
        return self._model

    @property
    def current_model_idx(self):
        return self._model_idx
