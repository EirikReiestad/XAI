import logging

import gymnasium as gym

from data_handler.src.data_handler import DataHandler
from environments.gymnasium.wrappers.cav_wrapper import CAVWrapper

env = gym.make("TagEnv-v0", render_mode="rgb_array")
env_wrapped = CAVWrapper(env)
concepts = env_wrapped.get_concepts()
logging.info(f"Concepts: {concepts}")

concept = "box-not-exist"

data_handler = DataHandler()
data_handler.generate_data(env_wrapped, concept=concept, n_samples=1000)
data_handler.save(f"{concept}.csv")
