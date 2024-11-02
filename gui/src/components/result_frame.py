import customtkinter as ctk

from .image_frame import ImageFrame


class ResultFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=3)

        self.result_viewer_frame = ResultViewerFrame(self)
        self.result_viewer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

        self.result_info_frame = ResultInfoFrame(self)

    def update_result(
        self,
        show_q_values: bool,
        show_tcav: bool = False,
        binary_concept_score: dict[str, float] | None = None,
        tcav_score: dict[str, float] | None = None,
    ):
        self.result_viewer_frame.update_result(show_q_values)
        self.result_info_frame.update_concept(
            show_tcav, binary_concept_score, tcav_score
        )

        if not show_tcav:
            self.result_info_frame.grid_forget()
        else:
            self.result_info_frame.grid(
                row=0, column=1, padx=10, pady=10, sticky="nswe"
            )


class ResultInfoFrame(ctk.CTkFrame):
    def __init__(self, master: ctk.CTkFrame | ctk.CTk):
        super().__init__(master)
        self.grid_columnconfigure((0, 1), weight=1)
        self.grid_rowconfigure((0, 1), weight=1)
        self.concept_viewer_frame = ConceptViewerFrame(self)
        self.concept_viewer_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nswe")

    def update_concept(
        self,
        show_tcav: bool,
        binary_concept_score: dict[str, float] | None,
        tcav_score: dict[str, float] | None,
    ):
        if not show_tcav or binary_concept_score is None or tcav_score is None:
            return
        self.concept_viewer_frame.update_concept(binary_concept_score, tcav_score)


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
        self.concept_infos = [ctk.CTkLabel(self, text="")]

    def update_concept(
        self, binary_concept_score: dict[str, float], tcav_score: dict[str, float]
    ):
        concept_infos = []
        for layer, score in tcav_score.items():
            concept_infos.append(f"{layer}: {score}")

        self.concept_infos = [
            ctk.CTkLabel(self, text=concept_info) for concept_info in concept_infos
        ]
        for i, concept_info in enumerate(self.concept_infos):
            concept_info.grid(row=i, column=0, padx=10, pady=10, sticky="nswe")


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
