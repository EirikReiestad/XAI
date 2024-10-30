import gymnasium as gym

from methods.src.cav import Analysis
from methods.src.utils import Models
from rl.src.dqn.policies import DQNPolicy


positive_concept = "seeker-next-to-hider"
negative_concept = "random1"

gym = gym.make("TagEnv-v0")
observation_space = gym.observation_space
action_space = gym.action_space

model = DQNPolicy(observation_space, action_space, [128, 128], [32, 32])
models = Models(model)

analysis = Analysis(models, positive_concept + ".csv", negative_concept + ".csv")
analysis.run(averages=5)
analysis.plot()
