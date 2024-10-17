import gymnasium as gym

from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy

project_folder = "tag-v0-eirre"
model_name = "tag-v0"
models = [
    "model_50:v76",
    "model_350:v72",
    "model_650:v69",
    #    "model_950:v54",
    #    "model_1250:v50",
    #    "model_1550:v42",
    #    "model_1850:v24",
    #    "model_2150:v20",
    #    "model_2450:v20",
    #    "model_2750:v19",
    #    "model_3050:v16",
]

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

model = DQNPolicy(observation_space, action_space, [128, 128], [])
models = Models(model, project_folder, model_name, models)
analysis = Analysis(models, "random.csv", "random.csv")
analysis.run()
analysis.plot()
