import logging

import gymnasium as gym

from models.src.download import ModelDownloader
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"
models0 = [
    "model_100:v456",
    "model_1000:v269",
    "model_5000:v85",
    "model_10000:v35",
    "model_15000:v24",
    "model_20100:v12",
    "model_25000:v10",
    "model_30000:v6",
]

models1 = [
    "model_100:v457",
    "model_1000:v270",
    "model_5000:v86",
    "model_10000:v36",
    "model_15000:v25",
    "model_20100:v13",
    "model_25000:v11",
    "model_30000:v7",
]

models0 = [
    "model_100:v518",
    "model_500:v393",
    "model_1000:v325",
    "model_1500:v246",
    "model_2000:v193",
    "model_2500:v188",
    "model_3000:v164",
    "model_3500:v150",
    "model_4000:v136",
]
models1 = [
    "model_100:v519",
    "model_500:v394",
    "model_1000:v326",
    "model_1500:v247",
    "model_2000:v194",
    "model_2500:v189",
    "model_3000:v165",
    "model_3500:v151",
    "model_4000:v137",
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
