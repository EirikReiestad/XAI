from environments.gymnasium.utils.enums import StateType

# Maze settings
MAZE_HEIGHT = 5
MAZE_WIDTH = 5

# Reward settings
GOAL_REWARD = 10
MOVE_REWARD = -1
TERMINATED_REWARD = -10
TRUNCATED_REWARD = -1

# Screen settings
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Path: environments
FILENAME = "maze-0-10-10.txt"

# State
STATE_TYPE = StateType.RGB
