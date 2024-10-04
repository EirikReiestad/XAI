import customtkinter as ctk
from .position_widget import PositionWidget
from gui.src.utils import EnvHandler


class ConfigurationFrame(ctk.CTkFrame):
    def __init__(self, master, env_handler: EnvHandler, update_result_callback):
        super().__init__(master)
        self.update_result_callback = update_result_callback

        self.grid_columnconfigure(0, weight=1)

        self.env_matrix_configuration_frame = EnvMatrixConfigurationFrame(
            self, env_handler
        )
        self.env_matrix_configuration_frame.grid(
            row=2, column=0, padx=10, pady=10, sticky="nse"
        )

        self.button = ctk.CTkButton(self, text="Update", command=self.update)
        self.button.grid(row=3, column=0, padx=10, pady=10, sticky="nswe")

    def update(self):
        self.update_result_callback(
            self.env_matrix_configuration_frame.get_env_matrix()
        )


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
