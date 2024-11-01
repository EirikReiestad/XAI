import logging

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from methods.src.cav import CAV
from methods.src.utils import Models


class Analysis:
    def __init__(
        self,
        models: Models,
        positive_sample_path: str,
        negative_sample_path: str,
    ) -> None:
        self._models = models
        self._positive_sample_path = positive_sample_path
        self._negative_sample_path = negative_sample_path

        self._cav_scores = {}
        self._tcav_scores = {}

        self._model_steps = {}

        self._total_cav_scores = {}
        self._total_tcav_scores = {}
        self._average_cav_scores = {}

    def run(self, averages: int = 10):
        for i in range(averages):
            self._reset()
            self._models.reset()
            logging.info(f"Running CAV {i + 1}/{averages}")
            self._run_cav()
            self._add_total_cav_scores()
            self._add_total_tcav_scores()

        self._calculate_average_cav_scores(averages)

    def _reset(self):
        self._cavs = {}
        self._binary_concept_scores = {}
        self._tcav_scores = {}
        np.random.seed(0)

    def _add_total_cav_scores(self):
        for model, layers in self._cav_scores.items():
            if model not in self._total_cav_scores:
                self._total_cav_scores[model] = {}
            for (
                layer,
                score,
            ) in layers.items():
                if layer not in self._total_cav_scores[model]:
                    self._total_cav_scores[model][layer] = 0
                self._total_cav_scores[model][layer] += score

    def _add_total_tcav_scores(self):
        for model, layers in self._tcav_scores.items():
            if model not in self._total_tcav_scores:
                self._total_tcav_scores[model] = {}
                for (
                    layer,
                    score,
                ) in layers.items():
                    if layer not in self._total_tcav_scores[model]:
                        self._total_tcav_scores[model][layer] = 0
                    self._total_tcav_scores[model][layer] += score

    def _calculate_average_cav_scores(self, n: int):
        for model, layers in self._total_cav_scores.items():
            self._average_cav_scores[model] = {}
            for (
                layer,
                score,
            ) in layers.items():
                self._average_cav_scores[model][layer] = score / n

    def _run_cav(self):
        while True:
            cav = CAV(
                self._models.policy_net,
                self._positive_sample_path,
                self._negative_sample_path,
            )
            cavs, binary_concept_scores, tcav_scores = cav.compute_cavs()
            self._cav_scores[self._models.current_model_idx] = (
                binary_concept_scores.copy()
            )
            self._tcav_scores[self._models.current_model_idx] = tcav_scores.copy()
            self._model_steps[self._models.current_model_idx] = (
                self._models.current_model_steps
            )
            if not self._models.has_next():
                break
            self._models.next()

    @property
    def cav_scores(self):
        return self._average_cav_scores.copy()

    @property
    def tcav_scores(self):
        return self._total_tcav_scores.copy()

    @property
    def steps(self):
        return self._model_steps.copy()

    @staticmethod
    def plot(
        scores: list[dict],
        steps: list[dict],
        folder_path: str = "assets/figures",
        filename: str = "cav_plot.png",
        labels: list[str] | None = None,
        title: str = "Plot",
        show: bool = True,
    ):
        matrices = []
        for score in scores:
            matrix = np.array([list(s.values()) for s in score.values()])
            matrices.append(np.array(matrix))

        use_labels = True
        if labels is None:
            use_labels = False
            labels = [str(i) for i in range(len(matrices))]

        assert len(labels) == len(matrices)
        save_path = f"{folder_path}/{filename}"

        model_steps = [f"{step:.1e}" for step in steps[0].values()]

        fig = plt.figure()
        ax1 = fig.add_subplot(projection="3d")

        _x = np.arange(matrices[0].shape[1])
        _y = np.arange(matrices[0].shape[0])
        _xx, _yy = np.meshgrid(_x, _y)

        for i, (matrix, label) in enumerate(zip(matrices, labels)):
            cmap = cm.get_cmap("viridis")
            color = cmap(i / len(matrices))
            ax1.plot_surface(
                _xx,
                _yy,
                matrix,
                edgecolor="k",
                color=color,
                alpha=0.5,
                shade=True,
                label=label,
            )
            if use_labels:
                ax1.text2D(
                    0.05,
                    0.05 - i * 0.05,
                    label,
                    transform=ax1.transAxes,
                    fontsize=10,
                    color=color,
                )

        ax1.set_zlim(0, 1)
        ax1.set_title(title)
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Steps")
        ax1.set_zlabel("Score")

        ax1.set_xticks(np.arange(matrices[0].shape[1]))
        ax1.set_xticklabels([str(i) for i in range(1, matrices[0].shape[1] + 1)])
        ax1.set_yticks(np.arange(len(model_steps)))
        ax1.set_yticklabels(model_steps)

        if show:
            plt.show()
        fig.savefig(save_path)
