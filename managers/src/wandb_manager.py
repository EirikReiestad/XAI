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
        self.sweep_config = self._get_sweep_config()

    def _get_sweep_config(self) -> dict:
        return {
            "method": "bayes",  # grid, random, bayes
            "metric": {
                "name": "agent0_average_reward",
                "goal": "maximize",
            },
            "parameters": {
                "learning_rate": {"values": [1e-4, 1e-3, 1e-2]},
                "gamma": {"values": [0.9, 0.95, 0.99]},
                "eps_start": {"values": [1.0, 0.9]},
                "eps_end": {"values": [0.1, 0.05, 0.01]},
                "eps_decay": {"values": [10000, 20000, 50000, 100000]},
                "batch_size": {"values": [16, 32, 64, 128]},
                "tau": {"values": [0.01, 0.005]},
                "hidden_layers": {
                    "values": [
                        [128, 128],
                        [128, 64, 32],
                        [256, 128],
                        [128, 64],
                        [128, 128, 64],
                        [256, 128, 64],
                    ]
                },
                "conv_layers": {"values": [[], [64, 32], [64, 64], [128, 64, 32]]},
                "memory_size": {"values": [10000, 50000]},
            },
        }


class WandBManager:
    initialized = False

    def __init__(self, active: bool, config: WandBConfig | None):
        if self.initialized:
            active = False
        self.active = active

        self.api = wandb.Api()

        if config is None:
            config = WandBConfig()
        self.config = config

        if not self.active:
            return

        self.reinit()

        self.cleanup_counter = 0
        WandBManager.initialized = True

    def reinit(self):
        wandb.init(
            project=self.config.project,
            name=self.config.run_name,
            config=self.config.other,
            reinit=True,
            mode="online",
            tags=self.config.tags,
            dir=self.config.dir,
        )

    def sweep(self) -> str:
        sweep_id = wandb.sweep(self.config.sweep_config, project=self.config.project)
        return sweep_id

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

        try:
            artifact_name = f"{model_artifact}_{step}"
            artifact = wandb.Artifact(artifact_name, type="model")
            artifact.metadata = metadata
            artifact.add_file(path)
            wandb.log_artifact(artifact)
            wandb.log({"model_logged": True}, step=step + 1)
        except wandb.Error as e:
            logging.error(
                f"Error: Could not save model with artifact: {model_artifact}"
            )
            logging.error(f"Error: {e}")
        except Exception as e:
            logging.error(
                f"Error: Could not save model with artifact: {model_artifact}"
            )
            logging.error(f"Error: {e}")

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
        if version_number == "":
            logging.error("Error: version_number cannot be empty")
        artifact_path = f"{run_path}/{model_artifact}:{version_number}"
        try:
            artifact = self.api.artifact(
                artifact_path,
            )
            artifact_dir = artifact.download()
            logging.info("model downloaded at: " + artifact_dir)
            logging.info("Metadata: " + str(artifact.metadata))
            return artifact_dir, artifact.metadata
        except wandb.Error as e:
            logging.error(f"Error: Could not load model with artifact: {artifact_path}")
            logging.error(f"Error: {e}")
            return None, None

    def save_gif(self, path: str, step: int, append: str = ""):
        if not self.active:
            return
        try:
            wandb.log({f"gif{append}": wandb.Image(path)}, step=step + 1)
        except Exception as e:
            logging.error(f"Error: Could not save gif at path: {path}")
            logging.error(f"Error: {e}")
