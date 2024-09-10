from demo.src.demos.demo_type import DemoType
from demo.src.demos.rl_type import RLType

DEMO_TYPE = DemoType.TAG
RL_TYPE = RLType.DQN

NUM_EPISODES = 100000
RENDER = True
RENDER_EVERY = 50

SLOWING_FACTOR = 100

SAVE_MODEL = True
SAVE_EVERY = 500
SAVE_MODEL_NAME = "model"

PRETRAINED = False
LOAD_MODEL_NAME = "2024-08-30_11-06-27/model_3400"

PLOT_AGENT_REWARD = True
PLOT_OBJECT_MOVEMENT = False

RENDER_Q_VALUES = True
