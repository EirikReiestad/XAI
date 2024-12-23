import logging

import gymnasium as gym

from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper
from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy
from methods.src.cav.plot import plot_3d, plot_bar_with_bootstrap, plot_line
from methods.src.cav.utils import difference, log_data

negative_concept = "random_negative"

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

env = CAVWrapper(gym)
concept_names = ["has-direct-sight"]
concept_names = ["box-not-exist", "random", "has-direct-sight"]
concept_names = ["box-not-exist"]
concept_names = ["random", "box-not-exist"]
concept_names = [
    "random",
    "has-direct-sight",
    "agents-close",
    "hider-not-exist",
    "seeker-not-exist",
]
concept_names = ["random", "seeker-not-exist"]
concept_names = env.get_concept_names()


def analyse(
    positive_concept: str, negative_concept: str, observation_space, action_space
):
    cav_scores = []
    tcav_scores = []
    steps = []
    prefix = ""

    plot_distribution = False
    for suffix in ["0", "1"]:
        model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])
        models = Models(model, folder_suffix=suffix)
        analysis = Analysis(
            models,
            positive_concept + ".csv",
            negative_concept + ".csv",
            plot_distribution=plot_distribution,
            scaler="",
        )
        prefix = analysis._scaler
        analysis.run(averages=20)
        cav_scores.append(analysis.cav_scores)
        tcav_scores.append(analysis.tcav_scores)
        steps.append(analysis.steps)

        plot_distribution = False

    return cav_scores, tcav_scores, steps, prefix


concept_cav_scores = {}
concept_tcav_scores = {}
concept_diff_cav_scores = {}
concept_diff_tcav_scores = {}
for concept in concept_names:
    logging.info(f"Running CAV for concept {concept}")
    cav_scores, tcav_scores, steps, prefix = analyse(
        concept, negative_concept, observation_space, action_space
    )
    concept_cav_scores[concept] = cav_scores
    concept_tcav_scores[concept] = tcav_scores

    plot_3d(
        cav_scores,
        steps,
        filename=f"{concept}{prefix}.png",
        labels=["seeker", "hider"],
        title=concept,
        show=False,
    )

    plot_3d(
        tcav_scores,
        steps,
        filename=f"{concept}{prefix}_tcav.png",
        labels=["seeker", "hider"],
        title=concept,
        show=False,
    )

    log_data(cav_scores[0], filename=f"{concept}{prefix}_seeker")
    log_data(cav_scores[1], filename=f"{concept}{prefix}_hider")

    cav_diff = difference(cav_scores[0], cav_scores[1])
    tcav_diff = difference(tcav_scores[0], tcav_scores[1])

    log_data(cav_diff, filename=f"{concept}{prefix}_diff")
    log_data(tcav_diff, filename=f"{concept}{prefix}_diff_tcav")

    concept_diff_cav_scores[concept] = [cav_diff]
    concept_diff_tcav_scores[concept] = [tcav_diff]

    plot_3d(
        [cav_diff],
        steps,
        filename=f"{concept}{prefix}_diff.png",
        labels=["difference(seeker - hider)"],
        title=concept,
        show=False,
        z_min=-1,
    )

    plot_3d(
        [tcav_diff],
        steps,
        filename=f"{concept}{prefix}_diff_tcav.png",
        labels=["difference(seeker - hider)"],
        title=concept,
        show=False,
        z_min=-1,
    )

log_data(concept_cav_scores, filename="cav_scores")
log_data(concept_tcav_scores, filename="tcav_scores")
log_data(concept_diff_cav_scores, filename="diff_cav_scores")
log_data(concept_diff_tcav_scores, filename="diff_tcav_scores")

plot_line(
    concept_cav_scores,
    title="Line plot",
    labels=["seeker", "hider"],
    show=False,
    filename="cav_line.png",
)
plot_line(
    concept_tcav_scores,
    title="TCAV Line plot",
    labels=["seeker", "hider"],
    show=False,
    filename="tcav_line.png",
)

plot_line(
    concept_diff_cav_scores,
    title="Line plot",
    labels=["difference(seeker - hider)"],
    show=False,
    filename="cav_diff_line.png",
    y_min=-1,
)

plot_line(
    concept_diff_tcav_scores,
    title="TCAV Line plot",
    labels=["difference(seeker - hider)"],
    show=False,
    filename="tcav_diff_line.png",
    y_min=-1,
)

plot_bar_with_bootstrap(
    concept_cav_scores,
    title="Bar plot",
    labels=["seeker", "hider"],
    show=False,
    filename="cav_bar.png",
)

plot_bar_with_bootstrap(
    concept_tcav_scores,
    title="TCAV Bar plot",
    labels=["seeker", "hider"],
    show=False,
    filename="tcav_bar.png",
)
