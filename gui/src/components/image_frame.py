import customtkinter


class ImageFrame(customtkinter.CTkFrame):
    def __init__(self, master: customtkinter.CTkFrame):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
