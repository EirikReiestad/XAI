import logging

import gymnasium as gym

from models.src.download import ModelDownloader
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"
models0 = [
    "model_100:v371",
    # "model_500:v262",
    "model_1000:v203",
    # "model_1500:v130",
    "model_2000:v83",
    # "model_2500:v78",
    "model_3000:v56",
    # "model_3500:v42",
    "model_4000:v34",
    # "model_4500:v30",
    "model_5000:v28",
    # "model_5500:v28",
    "model_6000:v27",
    # "model_6500:v28",
    "model_7000:v28",
    # "model_7500:v18",
    "model_8000:v18",
    # "model_8500:v14",
    "model_9000:v16",
    # "model_9400:v16",
]

models1 = [
    "model_100:v372",
    # "model_500:v263",
    "model_1000:v204",
    # "model_1500:v131",
    "model_2000:v84",
    # "model_2500:v79",
    "model_3000:v57",
    # "model_3500:v43",
    "model_4000:v35",
    # "model_4500:v31",
    "model_5000:v29",
    # "model_5500:v29",
    "model_6000:v28",
    # "model_6500:v29",
    "model_7000:v29",
    # "model_7500:v19",
    "model_8000:v19",
    # "model_8500:v15",
    "model_9000:v17",
    # "model_9400:v17",
]

models = models1

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])

model_downloader = ModelDownloader(
    model, project_folder, model_name, models, folder_suffix="1"
)
logging.info("Downloading models")
model_downloader.download()
