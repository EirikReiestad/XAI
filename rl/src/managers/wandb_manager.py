import wandb
import os
from dataclasses import dataclass
import logging


@dataclass
class WandBConfig:
    def __init__(
        self,
        project: str = "",
        run_name: str = "",
        tags: list[str] = [],
        other: dict = {},
    ) -> None:
        self.project: str = project
        self.run_name: str = run_name
        self.tags: list[str] = tags
        self.other: dict = other


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
            tags=config.tags,
        )

    def log(self, data: dict):
        if not self.active:
            return
        wandb.log(data)

    def finish(self):
        if not self.active:
            return
        wandb.finish()

    def save_model(self, path: str, step: int, model_artifact: str = "model") -> None:
        if not self.active:
            return

        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            logging.warning(f"Error: The file {path} does not exist or is empty.")
            return

        artifact_name = f"{model_artifact}_{step}"
        artifact = wandb.Artifact(artifact_name, type="model")
        artifact.add_file(path)
        wandb.log_artifact(artifact)
        wandb.log({"model_logged": True}, step=step + 1)

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
