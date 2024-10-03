import customtkinter
import numpy as np

from utils.src.tag_generator.utils import DrawMode

from .components import ConfigurationFrame, ResultFrame
from .utils import EnvHandler, ModelHandler


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.env_handler = EnvHandler(10, 10)
        self.env_handler.generate()

        self.model_handler = ModelHandler("model_3000", ["v0", "v1"])
        self.model_handler.generate_shap()

        self.title("")
        self.geometry("800x800")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.result_frame = ResultFrame(self, self.env_handler)
        self.result_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.configuration_frame = ConfigurationFrame(
            self, update_result_callback=self.update_result
        )
        self.configuration_frame.grid(
            row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nswe"
        )

    def update_result(
        self,
        seeker_position: tuple[int, int] | None = None,
        hider_position: tuple[int, int] | None = None,
    ):
        self._update_seeker_position(seeker_position)
        self._update_hider_position(hider_position)
        self.result_frame.update_result()
        self.model_handler.generate_shap(np.array(self.env_handler.env))

    def _update_seeker_position(self, position: tuple[int, int] | None):
        if position is None:
            return
        self.env_handler.current_square = position
        self.env_handler.placement_mode = DrawMode.SEEKER
        self.env_handler.generate()

    def _update_hider_position(self, position: tuple[int, int] | None):
        if position is None:
            return
        self.env_handler.current_square = position
        self.env_handler.placement_mode = DrawMode.HIDER
        self.env_handler.generate()
