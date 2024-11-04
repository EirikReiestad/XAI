import logging

import gymnasium as gym

from models.src.download import ModelDownloader
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"

models0 = [
    "model_100:v434",
    "model_1000:v229",
    "model_2000:v111",
    "model_3000:v86",
    "model_4000:v34",
    "model_5000:v62",
    "model_6000:v56",
    "model_7000:v52",
    "model_8000:v40",
    "model_9000:v32",
    "model_10000:v19",
    "model_11000:v24",
    "model_12000:v22",
    "model_13000:v22",
]

models0 = [
    "model_100:v434",
    "model_300:v401",
    "model_500:v310",
    "model_700:v292",
    "model_900:v267",
]

models1 = [
    "model_100:v435",
    "model_300:v402",
    "model_500:v311",
    "model_700:v293",
    "model_900:v268",
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
