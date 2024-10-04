import customtkinter as ctk

from .image_frame import ImageFrame


class ResultFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

        self.env_viewer_frame = EnvViewerFrame(self)
        self.env_viewer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.shap_viewer_frame = ShapViewerFrame(self)
        self.shap_viewer_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")

    def update_result(
        self,
    ):
        self.env_viewer_frame.update_image()
        self.shap_viewer_frame.update_image()


class EnvViewerFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkFrame,
    ):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.image_frame = ImageFrame(self, "image.png")
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

    def update_image(self):
        self.image_frame.update_image("image.png")


class ShapViewerFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkFrame,
    ):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.image_frame_0 = ImageFrame(self, "0shap.png")
        self.image_frame_0.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.image_frame_1 = ImageFrame(self, "1shap.png")
        self.image_frame_1.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")

    def update_image(self):
        self.image_frame_0.update_image("0shap.png")
        self.image_frame_1.update_image("1shap.png")
