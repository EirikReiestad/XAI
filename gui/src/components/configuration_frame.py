import customtkinter as ctk
from .position_widget import PositionWidget


class ConfigurationFrame(ctk.CTkFrame):
    def __init__(self, master, update_result_callback):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.seeker_position_widget = PositionWidget(self, "Seeker")
        self.seeker_position_widget.grid(
            row=0, column=0, padx=10, pady=10, sticky="nswe"
        )
        self.hider_position_widget = PositionWidget(self, "Hider")
        self.hider_position_widget.grid(
            row=1, column=0, padx=10, pady=10, sticky="nswe"
        )

        self.button = ctk.CTkButton(self, text="Update", command=self._update)
        self.button.grid(row=2, column=0, padx=10, pady=10, sticky="nswe")

        self.update_result_callback = update_result_callback

    def _update(self):
        self.update_result_callback(
            self.seeker_position_widget.get_position(),
            self.hider_position_widget.get_position(),
        )
