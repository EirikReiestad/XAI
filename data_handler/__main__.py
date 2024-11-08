import logging

import gymnasium as gym

from data_handler.src.data_handler import DataHandler
from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper

env = gym.make("TagEnv-v0", render_mode="rgb_array")
env_wrapped = CAVWrapper(env)
concepts = env_wrapped.get_concepts()
logging.info(f"Concepts: {concepts}")
concept_names = ["random"]
concept_names = env_wrapped.get_concept_names()

for concept in concept_names:
    logging.info(f"{'='*5} Generating data for concept: {concept}{'='*5}")
    data_handler = DataHandler()
    data_handler.generate_data(env_wrapped, concept=concept, n_samples=5000)
    data_handler.show_random_sample(4)
    data_handler.save(f"{concept}.csv")
