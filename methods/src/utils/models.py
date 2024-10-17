import getpass
import traceback

import torch

from managers import WandBConfig, WandBManager
from rl.src.dqn.policies import DQNPolicy


class Models:
    def __init__(
        self,
        model: DQNPolicy,
        project_folder: str,
        model_name: str,
        models: list[str],
    ):
        current_user = getpass.getuser()
        project = f"{current_user}"
        wandb_config = WandBConfig(project=project)
        self.wandb_manager = WandBManager(active=False, config=wandb_config)

        self._model_name = model_name
        self._run_id = f"eirikreiestad-ntnu/{project_folder}"
        self._extract_model_names(models)
        self._model_idx = 0

        self._model = model
        self.next()

    def _extract_model_names(self, model_names: list[str]):
        model_artifacts = []
        version_numbers = []

        for model_name in model_names:
            split = model_name.split(":")
            model_artifacts.append(split[0])
            version_numbers.append(split[1])

        self._model_artifacts = model_artifacts
        self._version_numbers = version_numbers

    def next(self):
        self._model_artifact = self._model_artifacts[self._model_idx]
        self._version_number = self._version_numbers[self._model_idx]
        self._load_model()
        self._model_idx += 1

    def _load_model(self):
        artifact_dir, metadata = self.wandb_manager.load_model(
            self._run_id, self._model_artifact, self._version_number
        )
        if artifact_dir is None or metadata is None:
            raise Exception(f"Model not found, {traceback.format_exc}")

        path = f"{artifact_dir}/{self._model_name}"
        if not path.endswith(".pt"):
            path += ".pt"

        self._model.policy_net.load_state_dict(torch.load(path, weights_only=True))
        self._model.target_net.load_state_dict(self._model.policy_net.state_dict())
        self._model.policy_net.eval()
        self._model.target_net.eval()

    def has_next(self):
        return self._model_idx < len(self._model_artifacts)

    def current_model_name(self):
        return self._model_artifact

    @property
    def policy_net(self):
        return self._model.policy_net

    @property
    def target_net(self):
        return self._model.target_net

    @property
    def model(self):
        return self._model
