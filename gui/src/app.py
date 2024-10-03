import customtkinter
from .components import ConfigurationFrame, ResultFrame
from .utils import EnvHandler


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.env_handler = EnvHandler(10, 10)
        self.env_handler.generate()

        self.title("")
        self.geometry("800x800")
        self.grid_rowconfigure((0, 1), weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.result_frame = ResultFrame(self, self.env_handler)
        self.result_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.configuration_frame = ConfigurationFrame(
            self, update_result_callback=self.result_frame.update_result
        )
        self.configuration_frame.grid(
            row=0, column=1, rowspan=2, padx=10, pady=10, sticky="nswe"
        )
