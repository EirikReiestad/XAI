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
    "model_100:v528",
    "model_300:v493",
    "model_500:v401",
    "model_700:v378",
    "model_1000:v333",
    "model_1300:v281",
    "model_1500:v254",
    "model_1700:v218",
    "model_2000:v203",
    "model_2300:v194",
    "model_2500:v194",
    "model_2700:v186",
    "model_3000:v170",
]
models1 = [
    "model_100:v529",
    "model_300:v494",
    "model_500:v402",
    "model_700:v379",
    "model_1000:v334",
    "model_1300:v282",
    "model_1500:v255",
    "model_1700:v219",
    "model_2000:v204",
    "model_2300:v195",
    "model_2500:v195",
    "model_2700:v187",
    "model_3000:v171",
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
