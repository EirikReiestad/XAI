import customtkinter as ctk

from gui.src.utils import EnvHandler
from utils.src.tag_generator.utils import DrawMode

from .image_frame import ImageFrame


class ResultFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk, env_handler: EnvHandler):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.env_handler = env_handler

        self.image_frame_0 = ImageFrame(self, "image.png")
        self.image_frame_0.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.image_frame_1 = ImageFrame(self, "image.png")
        self.image_frame_1.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")

    def update_result(
        self,
        seeker_position: tuple[int, int] | None = None,
        hider_position: tuple[int, int] | None = None,
    ):
        self._update_seeker_position(seeker_position)
        self._update_hider_position(hider_position)

        self.env_handler.save_image("gui/src/assets/image.png")

    def _update_seeker_position(self, position: tuple[int, int] | None):
        print("Updating seeker")
        if position is None:
            return
        self.env_handler.current_square = position
        self.env_handler.draw_mode = DrawMode.SEEKER
        self.env_handler.generate()

    def _update_hider_position(self, position: tuple[int, int] | None):
        print("Updating hider")
        if position is None:
            return
        self.env_handler.current_square = position
        self.env_handler.draw_mode = DrawMode.SEEKER
        self.env_handler.generate()
