import logging

import customtkinter as ctk

from gui.src.utils import EnvHandler

from .image_frame import ImageFrame


class ConfigurationFrame(ctk.CTkFrame):
    def __init__(
        self,
        master,
        env_handler: EnvHandler,
        update_result_callback,
        update_image_callback,
    ):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.update_result_callback = update_result_callback
        self.update_image_callback = update_image_callback

        self.env_viewer_frame = EnvViewerFrame(self)
        self.env_viewer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

        self.env_matrix_configuration_frame = EnvMatrixConfigurationFrame(
            self, env_handler
        )
        self.env_matrix_configuration_frame.grid(
            row=1, column=0, padx=10, pady=10, sticky="nse"
        )

        self.model_chooser_frame = ModelChooserFrame(self)
        self.model_chooser_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nswe")

        self.shap_configuration_frame = ShapConfigurationFrame(self)
        self.shap_configuration_frame.grid(
            row=3, column=0, padx=10, pady=10, sticky="nswe"
        )

        self.checkbox = ctk.CTkCheckBox(
            self,
            text="Show Q-Values",
            command=self.update_image,
            variable=ctk.BooleanVar(value=True),
        )
        self.checkbox.grid(row=4, column=0, padx=10, pady=10, sticky="nswe")

        self.button = ctk.CTkButton(self, text="Update", command=self.update)
        self.button.grid(row=5, column=0, padx=10, pady=10, sticky="nswe")

    def update(self):
        self.update_result_callback(
            self.env_matrix_configuration_frame.get_env_matrix(),
            *self.model_chooser_frame.get_model(),
            self.shap_configuration_frame.get_shap_samples(),
            show_q_values=self.checkbox.get(),
        )
        self.env_viewer_frame.update_image()

    def update_image(self):
        self.update_image_callback(
            show_q_values=self.checkbox.get(),
        )


class ShapConfigurationFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkFrame,
    ):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.label = ctk.CTkLabel(self, text="SHAP Samples")
        self.shap_entry = ctk.CTkEntry(self, placeholder_text="10")
        self.shap_entry.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

    def get_shap_samples(self) -> int:
        if self.shap_entry.get() == "":
            value = self.shap_entry._placeholder_text
            if value == "" or value is None:
                logging.error("SHAP Samples is empty")
                return 10
            try:
                return int(value)
            except ValueError:
                logging.error("SHAP Samples is not an integer")
        return int(self.shap_entry.get())


class ModelChooserFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkFrame,
    ):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.model_entry = ctk.CTkEntry(
            self, placeholder_text="model artifact", width=40
        )
        self.model_entry.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

        self.x_entry = ctk.CTkEntry(self, placeholder_text="v0", width=40)
        self.x_entry.grid(row=0, column=1, sticky="nswe", padx=10, pady=10)
        self.y_entry = ctk.CTkEntry(self, placeholder_text="v1", width=40)
        self.y_entry.grid(row=0, column=2, sticky="nswe", padx=10, pady=10)

    def get_model(self) -> tuple[str, list[str]]:
        return self.model_entry.get(), [self.x_entry.get(), self.y_entry.get()]


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


class EnvMatrixConfigurationFrame(ctk.CTkFrame):
    def __init__(self, master, env_handler: EnvHandler):
        super().__init__(master)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        env = env_handler.env

        self.init_matrix = False

        self.matrix = []
        for i, row in enumerate(env):
            self.matrix.append([])
            for j, value in enumerate(row):
                color = self._get_color(value)
                validate_cmd = (self.register(self._validate_input), "%P", i, j)
                entry = ctk.CTkEntry(
                    self,
                    placeholder_text=str(value),
                    width=40,
                    fg_color=color,
                    text_color="black",
                    validate="key",
                    validatecommand=validate_cmd,
                )
                entry.grid(row=i, column=j, padx=1, pady=1)
                self.matrix[i].append(entry)

        self.init_matrix = True

        assert len(self.matrix) == len(env)
        assert len(self.matrix[0]) == len(env[0])

    def _validate_input(self, value, row: int, col: int) -> bool:
        if not self.init_matrix:
            return True

        row = int(row)
        col = int(col)

        if value.isdigit():
            self.matrix[row][col].configure(fg_color=self._get_color(int(value)))
            return True
        elif value == "":
            self.matrix[row][col].configure(fg_color=self._get_color(int(0)))
            return True
        else:
            self.matrix[row][col].configure(fg_color="red")
            return False

    def _get_color(self, value):
        if value == 0:
            return "white"
        elif value == 1:
            return "black"
        elif value == 2:
            return "blue"
        elif value == 3:
            return "green"
        elif value == 4:
            return "yellow"
        else:
            return "gray"

    def get_env_matrix(self):
        value_matrix = []
        for row in self.matrix:
            value_row = []
            for entry in row:
                value_row.append(
                    int(entry.get() if entry.get() != "" else entry._placeholder_text)
                )
            value_matrix.append(value_row)
        return value_matrix
