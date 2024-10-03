import customtkinter as ctk

from gui.src.utils import EnvHandler

from .image_frame import ImageFrame


class ResultFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk, env_handler: EnvHandler):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.env_handler = env_handler

        self.image_frame_0 = ImageFrame(self, "image.png")
        self.image_frame_0.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.image_frame_1 = ImageFrame(self, "shap.png")
        self.image_frame_1.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")

    def update_result(
        self,
    ):
        self.image_frame_0.update_image("image.png")
        self.image_frame_1.update_image("shap.png")
