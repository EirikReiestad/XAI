import unittest
from unittest.mock import patch, mock_open
from utils.src.maze_generator.utils import MazeDrawMode as DrawMode
from utils.src.maze_generator.generate_maze import GenerateMaze
from environments.gymnasium.envs.maze.utils import TileType
from environments.gymnasium.utils import Direction


class TestGenerateMaze(unittest.TestCase):
    @patch("pygame.display.set_mode")
    @patch("pygame.font.Font")
    def setUp(self, mock_font, mock_set_mode):
        self.maze = GenerateMaze(10, 10)

    def test_init(self):
        self.assertEqual(self.maze.width, 10)
        self.assertEqual(self.maze.height, 10)
        self.assertEqual(self.maze.draw_mode, DrawMode.NOTHING)
        self.assertEqual(self.maze.placement_mode, DrawMode.NOTHING)
        self.assertEqual(self.maze.current_square, (0, 0))

    @patch("utils.src.maze_generator.generate_maze.GenerateMaze._save_maze")
    def test_load_maze_new(self, mock_save):
        with patch("os.path.exists", return_value=False):
            self.maze._load_maze(0)
        self.assertEqual(len(self.maze.maze), 10)
        self.assertEqual(len(self.maze.maze[0]), 10)
        mock_save.assert_called_once()

    @patch(
        "builtins.open",
        new_callable=mock_open,
        read_data="0000000000\n" * 10,
    )
    def test_load_maze_existing(self, mock_file):
        with patch("os.path.exists", return_value=True):
            self.maze._load_maze(0)
        self.assertEqual(self.maze.maze, [[0] * 10 for _ in range(10)])

    def test_update_tile(self):
        self.maze._update_tile(DrawMode.OBSTACLE)
        self.assertEqual(self.maze.maze[0][0], TileType.OBSTACLE.value)

        self.maze._update_tile(DrawMode.ERASE)
        self.assertEqual(self.maze.maze[0][0], TileType.EMPTY.value)

    def test_update_placement(self):
        self.maze.placement_mode = DrawMode.START
        self.maze._update_placement()
        self.assertEqual(self.maze.maze[0][0], TileType.START.value)

        self.maze.current_square = (1, 1)
        self.maze.placement_mode = DrawMode.END
        self.maze._update_placement()
        self.assertEqual(self.maze.maze[1][1], TileType.END.value)

        # Using y, x
        self.maze.current_square = (1, 0)
        self.maze.placement_mode = DrawMode.START
        self.maze._update_placement()
        self.assertEqual(self.maze.maze[0][1], TileType.START.value)
        self.assertEqual(self.maze.maze[0][0], TileType.EMPTY.value)

    def test_move_current_square(self):
        self.maze.action = Direction.RIGHT
        self.maze._move_current_square()
        self.assertEqual(self.maze.current_square, (1, 0))

        self.maze.action = Direction.DOWN
        self.maze._move_current_square()
        self.assertEqual(self.maze.current_square, (1, 1))

    def test_get_color(self):
        self.assertEqual(self.maze._get_color(TileType.EMPTY.value), (255, 255, 255))
        self.assertEqual(self.maze._get_color(TileType.OBSTACLE.value), (0, 0, 0))
        self.assertEqual(self.maze._get_color(TileType.START.value), (0, 255, 0))
        self.assertEqual(self.maze._get_color(TileType.END.value), (255, 0, 0))


if __name__ == "__main__":
    unittest.main()
