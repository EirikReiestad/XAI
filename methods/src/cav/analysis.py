import logging

from methods.src.cav import CAV
from methods.src.utils import Models


class Analysis:
    def __init__(
        self,
        models: Models,
        positive_sample_path: str,
        negative_sample_path: str,
        plot_distribution: bool = False,
        scaler: str = "",
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
        self._average_tcav_scores = {}

        self._plot_distribution = plot_distribution

        self._scaler = scaler

    def run(self, averages: int = 10):
        for i in range(averages):
            self._reset()
            self._models.reset()
            logging.info(f"Running CAV {i + 1}/{averages}")
            self._run_cav()
            self._add_total_cav_scores()
            self._add_total_tcav_scores()

        self._calculate_average_cav_scores(averages)
        self._calculate_average_tcav_scores(averages)

    def _reset(self):
        self._cavs = {}
        self._binary_concept_scores = {}
        self._tcav_scores = {}

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

    def _calculate_average_tcav_scores(self, n: int):
        for model, layers in self._total_tcav_scores.items():
            self._average_tcav_scores[model] = {}
            for (
                layer,
                score,
            ) in layers.items():
                self._average_tcav_scores[model][layer] = score / n

    def _run_cav(self):
        plot_distribution = False
        while True:
            if not self._models.has_next():
                plot_distribution = self._plot_distribution
            cav = CAV(
                self._models.policy_net,
                self._positive_sample_path,
                self._negative_sample_path,
                self._scaler,
            )
            cavs, binary_concept_scores, tcav_scores = cav.compute_cavs(
                plot_distribution=plot_distribution
            )
            plot_distribution = False
            name = str(self._models.current_model_steps)
            self._cav_scores[name] = binary_concept_scores.copy()
            self._tcav_scores[name] = tcav_scores.copy()
            self._model_steps[name] = self._models.current_model_steps
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
