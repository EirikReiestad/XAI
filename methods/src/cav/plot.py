import math

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def plot_3d(
    scores: list[dict],
    steps: list[dict],
    folder_path: str = "assets/figures",
    filename: str = "cav_plot.png",
    labels: list[str] | None = None,
    title: str = "Plot",
    show: bool = True,
):
    save_path = f"{folder_path}/{filename}"
    matrices = []
    for score in scores:
        matrix = np.array([list(s.values()) for s in score.values()])
        matrices.append(np.array(matrix))

    if labels is None:
        labels = [str(i) for i in range(len(matrices))]

    assert len(labels) >= len(matrices) == len(scores)
    labels = labels[: len(matrices)]

    model_steps = [str(step) for step in steps[0].values()]
    score_labels = list(scores[0][model_steps[0]].keys())
    score_labels = [str(i) for i in range(1, len(score_labels) + 1)]

    model_steps = [step for step in steps[0].values()]
    formatted_steps = [
        (
            lambda step: f"{math.ceil(step / 10**int(math.log10(step))) if math.ceil(step / 10**int(math.log10(step))) < 10 else 1} * 10^{int(math.log10(step)) + (1 if math.ceil(step / 10**int(math.log10(step))) == 10 else 0)}"
        )(step)
        for step in model_steps
    ]
    formatted_steps = [f"10^{int(math.log10(step))}" for step in model_steps]

    n_plots = len(matrices)
    n_cols = n_plots
    n_rows = 1

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(20, 10), subplot_kw={"projection": "3d"}
    )

    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes)

    for i, (matrix, label) in enumerate(zip(matrices, labels)):
        ax = axes[i]

        _x = np.arange(matrices[0].shape[1])
        _y = np.arange(matrices[0].shape[0])
        _xx, _yy = np.meshgrid(_x, _y)

        cmap = cm.get_cmap("plasma")
        alpha = 1.0
        ax.plot_surface(
            _xx,
            _yy,
            matrix,
            edgecolor="k",
            cmap=cmap,
            alpha=alpha,
            shade=True,
        )

        # Change font size
        ax.tick_params(axis="both", which="major", labelsize=16)
        ax.set_zlim(0, 1)
        ax.set_title(f"{title} - {label}", fontsize=30)
        ax.set_xlabel("Layer", fontsize=20)
        ax.set_ylabel("Steps", fontsize=20, labelpad=20)
        ax.set_zlabel("Score", fontsize=20)
        ax.set_xticks(np.arange(matrices[0].shape[1]))
        ax.set_xticklabels(score_labels)
        ax.set_yticks(np.arange(len(model_steps)))
        ax.set_yticklabels(formatted_steps)

    fig.tight_layout()

    if show:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)


def plot_line(
    data: dict,
    folder_path: str = "assets/figures",
    filename: str = "line_plot.png",
    labels: list[str] | None = None,
    title: str = "Plot",
    show: bool = True,
):
    save_path = f"{folder_path}/{filename}"

    concept_matrices: list[dict] = [{} for _ in range(len(next(iter(data.values()))))]
    for concept, values in data.items():
        for i, value in enumerate(values):
            array = list(list(value.values())[-1].values())
            concept_matrices[i][concept] = array
            continue
            matrix = np.array([list(s.values()) for s in value.values()])
            average_matrix = np.mean(matrix, axis=0)
            concept_matrices[i][concept] = average_matrix

    concepts = [list(matrix.keys()) for matrix in concept_matrices]

    if labels is None:
        labels = [str(i) for i in range(len(concept_matrices))]

    assert len(labels) >= len(concept_matrices)
    labels = labels[: len(concept_matrices)]

    n_plots = len(concept_matrices)
    n_cols = n_plots
    n_rows = 1

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(20, 10),
    )
    plt.style.use("_mpl-gallery")

    if n_plots == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes)

    for i, (matrix, label, concept) in enumerate(
        zip(concept_matrices, labels, concepts)
    ):
        ax = axes[i]

        for concept, values in matrix.items():
            _x = np.arange(len(values))
            ax.plot(_x, values, marker="o", label=concept)

        ax.set_ylim(0, 1)
        ax.set_title(f"{title} - {label}")
        ax.set_xticks(np.arange(len(matrix[concept])))
        ax.set_xlabel("Layer")
        ax.set_ylabel("Average Score")
        if i == len(concept_matrices) - 1:
            ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Concepts")

    fig.tight_layout()

    if show:
        plt.show()
    fig.savefig(save_path)
    plt.close(fig)


def plot_bar_with_bootstrap(
    data: dict,
    n_bootstrap: int = 1000,
    conf_interval: float = 0.95,
    labels: list[str] | None = None,
    folder_path: str = "assets/figures",
    filename: str = "bar_conf.png",
    title: str = "Plot",
    show: bool = True,
):
    bootstrapped_means = {}

    for concept, value_list in data.items():
        concept_means = []
        concept_lower = []
        concept_upper = []

        for values in value_list:
            values = list(list(values.values())[-1].values())

            means = []
            for _ in range(n_bootstrap):
                sample = np.random.choice(values, size=len(values), replace=True)
                means.append(np.mean(sample))
                # sum_layer_values = [np.mean(list(s.values())) for s in sample]
                # means.append(np.mean(sum_layer_values))

            lower = np.percentile(means, (1 - conf_interval) / 2 * 100)
            upper = np.percentile(means, (1 + conf_interval) / 2 * 100)
            mean = np.mean(means)

            concept_means.append(mean)
            concept_lower.append(lower)
            concept_upper.append(upper)

        bootstrapped_means[concept] = {
            "means": concept_means,
            "lower": concept_lower,
            "upper": concept_upper,
        }

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (concept, values) in enumerate(bootstrapped_means.items()):
        means = values["means"]
        lower = values["lower"]
        upper = values["upper"]

        ax.bar(
            [
                f"{concept} - {labels[i] if labels is not None and len(labels) > i else i}"
                for i in range(len(means))
            ],
            means,
            yerr=[np.array(means) - np.array(lower), np.array(upper) - np.array(means)],
            capsize=5,
            label=concept,
        )

    ax.set_xlabel("Concepts")
    ax.set_ylabel("Mean Score")
    ax.set_ylim(0, 1)
    ax.set_title("Bar Plot with Bootstrapped 95% Confidence Intervals")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.title(title)
    ax.legend()  # Add a legend

    if show:
        plt.show()
    save_path = f"{folder_path}/{filename}"
    fig.savefig(save_path)
    plt.close(fig)
