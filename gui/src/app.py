import customtkinter
import numpy as np

from .components import ConfigurationFrame, ResultFrame
from .utils import EnvHandler, ModelHandler
from methods.src.cav import CAV


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        self.title("")
        self.geometry("1600x1200")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure(0, weight=2)
        self.grid_columnconfigure(1, weight=1)

        self.env_handler = EnvHandler(10, 10)
        self.env_handler.generate()

        self.model_handler = ModelHandler("model_20", ["v0", "v1"], shap_samples=10)
        self.cav = CAV(
            self.model_handler.dqn,
            "data/positive_samples.npy",
            "data/negative_samples.npy",
        )

        state = np.expand_dims(np.array(self.env_handler.env), axis=0)
        self.model_handler.generate_shap(state)
        self.result_frame = ResultFrame(self)
        self.result_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.configuration_frame = ConfigurationFrame(
            self,
            self.env_handler,
            update_result_callback=self.update_result,
            update_image_callback=self.update_image,
        )
        self.configuration_frame.grid(row=0, column=2, padx=10, pady=10, sticky="nswe")

    def update_result(
        self,
        env: list[list] | None,
        model_artifact: str,
        model_version_numbers: list[str],
        shap_samples: int,
        show_q_values: bool,
    ):
        self._update_env(env)
        state = np.expand_dims(np.array(self.env_handler.env), axis=0)

        if show_q_values:
            self.model_handler.update_model(model_artifact, model_version_numbers)
            self.model_handler.generate_q_values(state)
        self.model_handler.update_shap(shap_samples)
        self.model_handler.generate_shap(state)
        self.result_frame.update_result(q_values=show_q_values)

    def update_image(
        self,
        show_q_values: bool,
    ):
        self.result_frame.update_result(q_values=show_q_values)

    def _update_env(self, env: list[list] | None):
        if env is None:
            return
        self.env_handler.env = env
        self.env_handler.generate()

    def on_closing(self):
        self.destroy()
        self.quit()
