import wandb
from dataclasses import dataclass


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

    def save_model(self, path: str, model_artifact: str = "model") -> None:
        if not self.active:
            return
        artifact = wandb.Artifact(model_artifact, type="model")
        artifact.add_file(path)

    def load_model(self, run_id: str, model_artifact: str) -> None | str:
        if not self.active:
            return
        artifact = wandb.use_artifact(f"{run_id}:{model_artifact}", type="model")
        artifact_dir = artifact.download()
        return artifact_dir
