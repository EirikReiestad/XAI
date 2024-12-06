import logging

import gymnasium as gym

from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper
from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy
from methods.src.cav.plot import plot_3d, plot_bar_with_bootstrap, plot_line

negative_concept = "random_negative"

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

env = CAVWrapper(gym)
concept_names = ["has-direct-sight"]
concept_names = ["box-not-exist", "random", "has-direct-sight"]
concept_names = env.get_concept_names()
concept_names = ["random", "box-not-exist"]
concept_names = ["box-not-exist"]
concept_names = ["random"]


def analyse(
    positive_concept: str, negative_concept: str, observation_space, action_space
):
    cav_scores = []
    tcav_scores = []
    steps = []
    prefix = ""

    plot_distribution = False
    for suffix in ["0"]:
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
        analysis.run(averages=1)
        cav_scores.append(analysis.cav_scores)
        tcav_scores.append(analysis.tcav_scores)
        steps.append(analysis.steps)

        plot_distribution = False

    return cav_scores, tcav_scores, steps, prefix


concept_cav_scores = {}
concept_tcav_scores = {}
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

plot_bar_with_bootstrap(
    concept_cav_scores,
    title="Bar plot",
    labels=["seeker", "hider"],
    show=False,
    filename="cav_bar.png",
)
plot_bar_with_bootstrap(
    concept_cav_scores,
    title="TCAV Bar plot",
    labels=["seeker", "hider"],
    show=False,
    filename="tcav_bar.png",
)
