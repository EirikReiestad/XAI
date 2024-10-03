import customtkinter
from .components import ImageFrame, ConfigurationFrame


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hello World")
        self.geometry("800x800")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.image_frame = ImageFrame(self, "image.jpg")
        self.image_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.image_frame = ImageFrame(self, "image.jpg")
        self.image_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")
        self.configuration_frame = ConfigurationFrame(self)
        self.configuration_frame.grid(
            row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nswe"
        )

    def button_callback(self):
        pass
