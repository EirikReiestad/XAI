import customtkinter as ctk

from .image_frame import ImageFrame


class ResultFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)

        self.result_viewer_frame = ResultViewerFrame(self)
        self.result_viewer_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")

        self.result_info_frame = ResultInfoFrame(self)
        self.result_info_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")

    def update_result(
        self,
        q_values: bool,
    ):
        self.result_viewer_frame.update_result(q_values)


class ResultInfoFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.concept_viewer_frame = ConceptViewerFrame(self)


class ResultViewerFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.q_values_viewer_frame = DoubleViewerFrame(
            self, "saliency_0.png", "saliency_1.png"
        )
        self.q_values_viewer_frame.grid(
            row=0, column=0, padx=10, pady=10, sticky="nswe"
        )
        self.shap_viewer_frame = DoubleViewerFrame(self, "0shap.png", "1shap.png")
        self.shap_viewer_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nswe")

    def update_result(self, q_values: bool):
        if q_values == 1:
            self.q_values_viewer_frame.grid(
                row=0, column=0, padx=10, pady=10, sticky="nswe"
            )
            self.q_values_viewer_frame.update_image()
        else:
            self.q_values_viewer_frame.grid_forget()
        self.shap_viewer_frame.update_image()


class ConceptViewerFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)

    def update_concept(self):
        pass


class DoubleViewerFrame(ctk.CTkFrame):
    def __init__(
        self,
        master: ctk.CTkFrame,
        image_0: str,
        image_1: str,
    ):
        super().__init__(master)
        self.image_0 = image_0
        self.image_1 = image_1
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.image_frame_0 = ImageFrame(self, self.image_0)
        self.image_frame_0.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")
        self.image_frame_1 = ImageFrame(self, self.image_1)
        self.image_frame_1.grid(row=0, column=1, padx=10, pady=10, sticky="nswe")

    def update_image(self):
        self.image_frame_0.update_image(self.image_0)
        self.image_frame_1.update_image(self.image_1)
