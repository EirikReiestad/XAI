import customtkinter as ctk


class PositionWidget(ctk.CTkFrame):
    def __init__(self, master, label: str):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.label = ctk.CTkLabel(self, text=label)
        self.label.grid(row=0, column=0, sticky="nswe")
        self.x_entry = ctk.CTkEntry(self, placeholder_text="X")
        self.x_entry.grid(row=0, column=1, sticky="nswe")
        self.y_entry = ctk.CTkEntry(self, placeholder_text="Y")
        self.y_entry.grid(row=0, column=2, sticky="nswe")
