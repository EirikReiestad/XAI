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

    def run(self):
        while self._models.has_next():
            cav = CAV(
                self._models.policy_net,
                self._positive_sample_path,
                self._negative_sample_path,
            )
            cav.compute_cavs()
            cav.compute_cav_scores()
            self._cav_scores[self._models.current_model_name] = cav.cav_scores
            self._models.next()

    def plot(self):
        matrix = np.array(
            [list(scores.values()) for scores in self._cav_scores.values()]
        )
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

        plt.show()
