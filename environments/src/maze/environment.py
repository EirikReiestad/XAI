import pygame
import numpy as np
from ..environment import Environment
from ..grid import Grid
from ..direction import Direction
from .agent import Agent
from .goal import Goal


class MazeEnvironment(Environment):
    def __init__(self,
                 name: str,
                 description: str,
                 width: int,
                 height: int,
                 goal_x: int,
                 goal_y: int,
                 start_x: int = 0,
                 start_y: int = 0,
                 maze_path: str = 'maze.txt'):
        super().__init__(name, description)
        if not isinstance(width, int):
            raise ValueError("Width must be an integer")
        if not isinstance(height, int):
            raise ValueError("Height must be an integer")
        if width <= 0:
            raise ValueError("Width must be greater than 0")
        if height <= 0:
            raise ValueError("Height must be greater than 0")

        self._rewards = {
            "goal": 1.0,
            "move": 0.0,
            "wall": 0.0,
        }

        self.start_x = start_x
        self.start_y = start_y

        self.grid = Grid(width, height)
        self._init_goal(goal_x, goal_y)
        self._load_maze(path=maze_path)
        self._init_agent(start_x, start_y)

        if self._rewards is None:
            raise ValueError("Rewards must not be None")
        if self.agent is None:
            raise ValueError("Agent must not be None")
        if self.grid is None:
            raise ValueError("Grid must not be None")
        if self.goal is None:
            raise ValueError("Goal must not be None")

        self.game_over = False

    @property
    def rewards(self):
        return self._rewards

    @rewards.setter
    def rewards(self, rewards: dict):
        assert isinstance(rewards, dict), "Rewards must be a dictionary"
        if "goal" not in rewards:
            raise ValueError(
                "The rewards dictionary must contain a key 'goal' with a value")
        if "wall" not in rewards:
            raise ValueError(
                "The rewards dictionary must contain a key 'wall' with a value")
        if "move" not in rewards:
            raise ValueError(
                "The rewards dictionary must contain a key 'move' with a value")
        self._rewards = rewards

    def _init_agent(self, start_x: int, start_y: int):
        """
        Initialize the agent
        """
        self.agent = Agent(start_x, start_y)

    def _init_goal(self, pos_x: int, pos_y: int):
        """
        Initialize the Goal
        """
        self.goal = Goal(pos_x, pos_y)

    def _save_maze(self, path: str = 'maze.txt'):
        """
        """
        with open(path, 'w') as f:
            f.write('\n'.join([''.join([str(cell)
                    for cell in row]) for row in self.maze]))

    def _load_maze(self, path: str = 'maze.txt'):
        """
        The maze is loaded from a file
        """
        with open(path, 'r') as f:
            maze = f.read()
            maze = maze.strip().split('\n')
            maze = [[int(cell) for cell in row] for row in maze]
            if len(maze) != self.grid.height:
                raise ValueError(
                    f'Maze height must be {self.grid.height}, not {len(maze)}')
            if all(len(row) != self.grid.width for row in maze):
                raise ValueError(
                    f'Maze width must be {self.grid.width}, not {len(maze[0])}')
            self.grid.cells = maze

    def _game_over(self, new_x: int, new_y: int) -> (bool, float):
        """
        Return:
            (bool, float): game over, reward
        """
        if new_x == self.goal.pos_x and new_y == self.goal.pos_y:
            return True, self.rewards["goal"]
        return False, 0

    def _valid_direction(self, direction: Direction) -> bool:
        if direction is None:
            return False

        assert isinstance(
            direction, Direction), f"Direction must be a Direction, not {type(direction)}"

        new_x = self.agent.pos_x
        new_y = self.agent.pos_y

        match direction:
            case Direction.UP:
                new_y -= 1
            case Direction.DOWN:
                new_y += 1
            case Direction.LEFT:
                new_x -= 1
            case Direction.RIGHT:
                new_x += 1

        if new_x < 0 or new_x >= self.grid.width:
            return False
        if new_y < 0 or new_y >= self.grid.height:
            return False
        if self.grid[new_y][new_x] == 1:
            return False
        return True

    def set_screen(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        self.screen = screen
        pygame.display.set_caption(self.name)

    def run(self):
        pass

    def reset(self):
        self._init_agent(self.start_x, self.start_y)
        self.game_over = False

    def render(self, screen: pygame.Surface):
        assert screen is not None, "Screen must not be None"
        # Add rendering logic here, e.g., drawing shapes or images
        cell_width = screen.get_width() / self.grid.width
        cell_height = screen.get_height() / self.grid.height

        self.grid.render_grid(screen, cell_width, cell_height, env='maze')
        self.agent.render(screen, cell_width, cell_height)
        self.goal.render(screen, cell_width, cell_height)

    def step(self, action: Direction = None, reward: float = None) -> (bool, float):
        """
        Action is per now a synonym for direction, but it can be extended to include more actions

        Parameters:
            action: Direction
            reward (optional): float
                Add a reward for a specific action outside the environment

        Return:
            bool: Game over
            float: Reward
        """
        # We will just have a text to direction mapping to not make it mandatory to use the Direction enum
        if isinstance(action, str):
            action = self._text_to_direction(action)

        if isinstance(action, int):
            action = self._int_to_direction(action)

        if isinstance(action, np.int64):
            action = self._int_to_direction(action.item())

        if not self._valid_direction(action):
            return False, self.rewards["wall"]

        if reward is not None:
            return self._move_agent(action, reward)
        else:
            return self._move_agent(action)

    def _move_agent(self, direction: Direction = None, reward: float = 0.0) -> (bool, float):
        """
        Parameters:
            direction: Direction
            reward: float
                This is optional if we want to add a reward for a specific action outside the environment
        Return:
            bool: game over
            float: reward
        """
        if not isinstance(direction, Direction):
            raise ValueError("Direction must be a Direction")
        if not isinstance(reward, float):
            raise ValueError("Reward must be a float")

        self.agent.move(direction)
        reward += self.rewards["move"]
        game, game_reward = self._game_over(self.agent.pos_x, self.agent.pos_y)
        return game, reward + game_reward

    def _text_to_direction(self, text: str) -> Direction:
        match text:
            case "UP":
                return Direction.UP
            case "DOWN":
                return Direction.DOWN
            case "LEFT":
                return Direction.LEFT
            case "RIGHT":
                return Direction.RIGHT
            case _:
                assert ValueError(text), f"Invalid direction: {text}"

    def _int_to_direction(self, number: int) -> Direction:
        direction = Direction(number)
        if direction in Direction:
            return direction

    def get_state(self) -> np.ndarray:
        state = np.zeros((self.grid.width, self.grid.height))
        return state

    def state_to_index(self) -> int:
        """
        Converts the current state to a unique index.

        Returns:
            int: The unique index representing the current state.
        """
        num_cells = self.grid.width * self.grid.height
        num_actions = len(Direction)

        # Calculate base index based on agent position
        state_index = (self.agent.x * num_cells + self.agent.y) * num_actions

        # Add offset based on goal position (assuming unique goal location)
        state_index += self.goal.x + self.goal.y * num_cells

        return state_index
