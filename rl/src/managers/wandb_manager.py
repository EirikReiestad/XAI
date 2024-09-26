import logging
import os
import shutil
from dataclasses import dataclass

import wandb


@dataclass
class WandBConfig:
    def __init__(
        self,
        project: str = "",
        run_name: str = "",
        tags: list[str] = [],
        other: dict = {},
        dir: str = "/tmp/",
    ) -> None:
        self.project: str = project
        self.run_name: str = run_name
        self.tags: list[str] = tags
        self.other: dict = other
        self.dir: str = dir


class WandBManager:
    def __init__(self, active: bool, config: WandBConfig | None):
        self.active = active
        if not active:
            return

        self.api = wandb.Api()

        if config is None:
            config = WandBConfig()
        self.config = config
        wandb.init(
            project=config.project,
            name=config.run_name,
            config=config.other,
            reinit=True,
            mode="online",
            tags=config.tags,
            dir=config.dir,
        )

    def log(self, data: dict):
        if not self.active:
            return
        wandb.log(data)

    def finish(self):
        if not self.active:
            return
        wandb.finish()

    def save_model(
        self, path: str, step: int, model_artifact: str = "model", metadata: dict = {}
    ) -> None:
        if not self.active:
            return

        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            logging.warning(f"Error: The file {path} does not exist or is empty.")
            return

        artifact_name = f"{model_artifact}_{step}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.metadata = metadata
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.log({"model_logged": True}, step=step + 1)

        self._delete_local_models()

    def _delete_local_models(self):
        run = wandb.run
        if run is None:
            logging.warning("Error: Could not find wandb run path")
            return
        dir = run.dir
        assert dir[-5:] == "files"
        run_file = dir[:-6]
        shutil.rmtree(run_file)

    def load_model(
        self, run_path: str, model_artifact: str, version_number: str
    ) -> tuple[None | str, None | dict]:
        if not self.active:
            return None, None
        if version_number == "":
            logging.error("Error: version_number cannot be empty")
        artifact_path = f"{run_path}/{model_artifact}:{version_number}"
        try:
            artifact = self.api.artifact(
                artifact_path,
            )
            artifact_dir = artifact.download()
            logging.info("model loaded")
            logging.info("Metadata: " + str(artifact.metadata))
            return artifact_dir, artifact.metadata
        except wandb.Error as e:
            logging.error(f"Error: Could not load model with artifact: {artifact_path}")
            logging.error(f"Error: {e}")
            return None, None

    def save_gif(self, path: str, step: int, append: str = ""):
        if not self.active:
            return
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            logging.warning(f"Error: The file {path} does not exist or is empty.")
            return
        wandb.log({f"gif{append}": wandb.Image(path)}, step=step + 1)
