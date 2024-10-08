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
        dir: str = ".",
        cleanup: bool = True,
        cleanup_period: int = 20,
    ) -> None:
        self.project: str = project
        self.run_name: str = run_name
        self.tags: list[str] = tags
        self.other: dict = other
        self.dir: str = dir
        self.cleanup: bool = cleanup
        self.cleanup_period = cleanup_period


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

        self.cleanup_counter = 0

    def log(self, data: dict, step: int | None = None):
        if not self.active:
            return
        if step is not None:
            wandb.log(data, step=step)
        else:
            wandb.log(data)

    def cleanup(self):
        if not self.active or not self.config.cleanup:
            return
        if self.cleanup_counter >= self.config.cleanup_period:
            self.cleanup_counter = 0
            self._delete_local_models()
            self._clean_cache()
            self._clean_local()
            return
        self.cleanup_counter += 1

    def finish(self):
        if not self.active:
            return
        wandb.finish()
        wandb_dir = self.config.dir + "/wandb"
        try:
            shutil.rmtree(wandb_dir)
        except FileNotFoundError:
            logging.warning(f"Error: Could not find wandb directory: {wandb_dir}")

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

    def _delete_local_models(self):
        run = wandb.run
        if run is None:
            logging.warning("Error: Could not find wandb run path")
            return
        dir = run.dir
        assert dir[-5:] == "files"
        run_file = dir[:-6]
        try:
            shutil.rmtree(run_file)
        except FileNotFoundError:
            logging.warning(f"Error: Could not find run file: {run_file}")

    def _clean_cache(self):
        # https://community.wandb.ai/t/wandb-artifact-cache-directory-fills-up-the-home-directory/5224
        # This is a workaround to clean the cache as wandb fills up the cache directory
        cache_dir = os.path.expanduser("~/.cache/wandb/artifacts/obj")
        if not os.path.exists(cache_dir):
            return
        try:
            logging.info(
                f"Cleaning cache: {cache_dir} {os.path.getsize(cache_dir) / 1024 / 1024} MB"
            )
            shutil.rmtree(cache_dir)
        except FileNotFoundError:
            logging.warning(f"Error: Could not find cache directory: {cache_dir}")

    def _clean_local(self):
        local_dir = os.path.expanduser("~/.local/share/wandb/artifacts/staging")
        if not os.path.exists(local_dir):
            return
        try:
            logging.info(
                f"cleaning local: {local_dir} {os.path.getsize(local_dir) / 1024 / 1024} MB"
            )
            shutil.rmtree(local_dir)
        except FileNotFoundError:
            logging.warning(f"Error: Could not find local directory: {local_dir}")

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
