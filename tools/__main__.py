from tools import settings
from tools.maze_generator.generate_maze import GenerateMaze

if __name__ == "__main__":
    maze = GenerateMaze(settings.MAZE_WIDTH, settings.MAZE_HEIGHT)
    maze.generate()
