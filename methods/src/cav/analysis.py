import logging

import matplotlib.pyplot as plt
import numpy as np

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
        self._model_steps = {}

    def run(self):
        while True:
            cav = CAV(
                self._models.policy_net,
                self._positive_sample_path,
                self._negative_sample_path,
            )
            cav.compute_cavs()
            cav.compute_cav_scores()
            self._cav_scores[self._models.current_model_name] = cav.cav_scores
            self._model_steps[self._models.current_model_name] = (
                self._models.current_model_steps
            )
            if not self._models.has_next():
                break
            self._models.next()

    def plot(self, save_path: str = "cav_plot.png"):
        matrix = np.array(
            [list(scores.values()) for scores in self._cav_scores.values()]
        )
        steps = [f"{step:.1e}" for step in self._model_steps.values()]
        if len(matrix) == 0:
            logging.error("No CAV scores found.")
            return
        matrix_flat = matrix.flatten()

        fig = plt.figure()
        ax1 = fig.add_subplot(projection="3d")

        _x = np.arange(matrix.shape[1])
        _y = np.arange(matrix.shape[0])
        _xx, _yy = np.meshgrid(_x, _y)
        x, y = _xx.ravel(), _yy.ravel()

        top = matrix_flat
        bottom = np.zeros_like(top)
        width = depth = 1

        ax1.bar3d(x, y, bottom, width, depth, top, shade=True)

        ax1.set_zlim(0, 1)
        ax1.set_title("CAV Scores")
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Steps")
        ax1.set_zlabel("CAV Score")

        ax1.set_xticks(np.arange(matrix.shape[1]))
        ax1.set_xticklabels([str(i) for i in range(1, matrix.shape[1] + 1)])
        ax1.set_yticks(np.arange(len(steps)))
        ax1.set_yticklabels(steps)

        plt.show()

        fig.savefig(save_path)
