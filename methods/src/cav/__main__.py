import logging

import gymnasium as gym

from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper
from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy

negative_concept = "random_negative"

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

env = CAVWrapper(gym)
concept_names = env.get_concept_names()
concept_names = ["random", "has-direct-sight", "seeker-close-to-box"]


def plot(positive_concept: str, scores: list, steps: list, prefix: str = ""):
    Analysis.plot(
        scores,
        steps,
        filename=f"{positive_concept}{prefix}.png",
        title=positive_concept,
        show=False,
    )


def analyse(
    positive_concept: str, negative_concept: str, observation_space, action_space
):
    cav_scores = []
    tcav_scores = []
    steps = []

    for suffix in ["0"]:
        model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])
        models = Models(model, folder_suffix=suffix)
        analysis = Analysis(
            models, positive_concept + ".csv", negative_concept + ".csv"
        )
        analysis.run(averages=1)
        cav_scores.append(analysis.cav_scores)
        tcav_scores.append(analysis.tcav_scores)
        steps.append(analysis.steps)

    return cav_scores, cav_scores, steps


for concept in concept_names:
    logging.info(f"Running CAV for concept {concept}")
    cav_scores, tcav_scores, steps = analyse(
        concept, negative_concept, observation_space, action_space
    )
    plot(concept, cav_scores, steps, prefix="")
