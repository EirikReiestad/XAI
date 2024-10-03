import customtkinter as ctk
from .position_widget import PositionWidget


class ConfigurationFrame(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.position_widget = PositionWidget(self, "Seeker")
        self.position_widget.grid(row=0, column=0, sticky="nswe")
