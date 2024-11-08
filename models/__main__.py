import logging

import gymnasium as gym

from models.src.download import ModelDownloader
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"
models0 = [
    "model_1000:v269",
    "model_5000:v85",
    "model_10000:v35",
    "model_15000:v24",
    "model_20100:v12",
    "model_25000:v10",
    "model_30000:v6",
]

models1 = [
    "model_1000:v270",
    "model_5000:v86",
    "model_10000:v36",
    "model_15000:v25",
    "model_20100:v13",
    "model_25000:v11",
    "model_30000:v7",
]

models = [models0, models1]

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

for i, model in enumerate(models):
    dqn_model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])

    model_downloader = ModelDownloader(
        dqn_model, project_folder, model_name, model, folder_suffix=str(i)
    )
    logging.info("Downloading models")
    model_downloader.download()
