import logging

import gymnasium as gym

from models.src.download import ModelDownloader
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"
models0 = [
    "model_100:v462",
    "model_1000:v277",
    "model_2000:v155",
    "model_3000:v126",
    "model_4000:v100",
    "model_5000:v91",
    "model_6000:v83",
    "model_7000:v77",
    "model_8000:v66",
    "model_9000:v56",
]

models1 = [
    "model_100:v463",
    "model_1000:v278",
    "model_2000:v156",
    "model_3000:v127",
    "model_4000:v101",
    "model_5000:v92",
    "model_6000:v84",
    "model_7000:v78",
    "model_8000:v67",
    "model_9000:v57",
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
