import customtkinter as ctk
from gui.src.utils import open_image


class ImageFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame, path: str):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.image = open_image(f"gui/src/assets/{path}")
        self.ctk_image = ctk.CTkImage(self.image)
        self.label = ctk.CTkLabel(self, image=self.ctk_image, text="")
        self.label.grid(row=0, column=0, sticky="nswe")
        self.bind("<Configure>", self._resize_image)

    def _resize_image(self, event):
        image_aspect_ratio = self.image.width / self.image.height
        new_width, new_height = event.width, event.height

        if new_width / new_height > image_aspect_ratio:
            new_width = int(new_height * image_aspect_ratio)
        else:
            new_height = int(new_width / image_aspect_ratio)

        resized_image = self.image.resize((new_width, new_height))
        self.ctk_image.configure(size=(new_width, new_height))
        self.ctk_image = ctk.CTkImage(resized_image, size=(new_width, new_height))
        self.label.configure(image=self.ctk_image)

    def update_image(self, path: str):
        self.image = open_image(f"gui/src/assets/{path}")
        self.ctk_image = ctk.CTkImage(self.image)
        self.label.configure(image=self.ctk_image)
