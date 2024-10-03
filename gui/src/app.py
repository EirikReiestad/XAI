import customtkinter
from .components.checkbox_frame import MyCheckboxFrame


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Hello World")
        self.geometry("400x400")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        self.button = customtkinter.CTkButton(
            self, text="my button", command=self.button_callback
        )

        self.checkbox_frame = MyCheckboxFrame(self, "Values", values=["a", "b", "c"])
        self.checkbox_frame.grid(row=0, column=0, padx=10, pady=(10, 0), sticky="nsw")
        self.checkbox_frame.configure(fg_color="transparent")
        self.checkbox_frame_0 = MyCheckboxFrame(self, "Values", values=["a", "b", "c"])
        self.checkbox_frame_0.grid(row=0, column=1, padx=10, pady=(10, 0), sticky="nsw")

        self.button = customtkinter.CTkButton(
            self, text="my button", command=self.button_callback
        )
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="ew")

    def button_callback(self):
        print(self.checkbox_frame.get())
        print(self.checkbox_frame_0.get())
