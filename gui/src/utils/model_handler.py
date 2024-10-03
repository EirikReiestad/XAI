import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from environments.gymnasium.wrappers import MultiAgentEnv
from methods import Shap
from rl.src.dqn.wrapper import MultiAgentDQN
from rl.src.managers import WandBConfig


class ModelHandler:
    def __init__(self, model_artifact: str, version_numbers: list[str]):
        self.num_agents = 2

        env = gym.make("TagEnv-v0", render_mode="rgb_array")
        model_name = "tag-v0"
        self.env = MultiAgentEnv(env)
        wandb_config = WandBConfig(project="gui")
        self.dqn = MultiAgentDQN(
            self.env,
            self.num_agents,
            "dqnpolicy",
            wandb=False,
            wandb_config=wandb_config,
            model_name=model_name,
            save_every_n_episodes=100,
            save_model=False,
            load_model=True,
            gif=False,
            run_path="eirikreiestad-ntnu/tag-v0-idun",
            model_artifact=model_artifact,
            version_numbers=version_numbers,
        )
        self.shap = Shap(self.env, self.dqn, samples=10)

    def generate_shap(
        self, state: np.ndarray | None = None, filename: str = "shap.png"
    ):
        self.shap.explain()
        if state is None:
            return
        shap_values = self.shap.shap_values(state)
        plt.figure()
        self.shap.plot(shap_values, show=False)
        plt.savefig(f"gui/src/assets/{filename}")
        plt.close()
