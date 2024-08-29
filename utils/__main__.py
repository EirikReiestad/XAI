from . import settings

# from tools.maze_generator.generate_maze import GenerateMaze
from .src.coop_generator.generate_env import GenerateEnv

if __name__ == "__main__":
    env = GenerateEnv(settings.MAZE_WIDTH, settings.MAZE_HEIGHT)
    env.generate()
