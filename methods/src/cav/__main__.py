import logging

import gymnasium as gym

from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper
from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy

negative_concept = "random0"

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

env = CAVWrapper(gym)
concept_names = env.get_concept_names()
concept_names.remove("random")
concept_names.append("random1")


def plot(positive_concept: str, negative_concept: str, observation_space, action_space):
    scores = []
    steps = []

    for suffix in ["0", "1"]:
        model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])
        models = Models(model, folder_suffix=suffix)
        analysis = Analysis(
            models, positive_concept + ".csv", negative_concept + ".csv"
        )
        analysis.run(averages=1)
        scores.append(analysis.scores)
        steps.append(analysis.steps)

    Analysis.plot(
        scores,
        steps,
        filename=f"{positive_concept}.png",
        title=positive_concept,
        show=False,
    )


for concept in concept_names:
    logging.info(f"Running CAV for concept {concept}")
    plot(concept, negative_concept, observation_space, action_space)
