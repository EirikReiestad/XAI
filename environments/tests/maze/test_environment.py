import unittest
from src.maze.environment import MazeEnvironment
from src.direction import Direction


class TestMazeEnvironment(unittest.TestCase):
    # create a test case
    with open('test_maze.txt', 'w') as f:
        f.write('0000\n')
        f.write('0111\n')
        f.write('0111\n')
        f.write('0000\n')

    def test_init(self):
        name = "Test Maze Environment"
        description = "Test description"
        width = 4
        height = 4
        env = MazeEnvironment(name, description, width,
                              height, goal_x=3, goal_y=3, start_x=0, start_y=0, maze_path='test_maze.txt')
        self.assertEqual(env.name, name)
        self.assertEqual(env.description, description)
        self.assertEqual(env.grid.width, width)
        self.assertEqual(env.grid.height, height)
        self.assertIsNotNone(env.agent)
        self.assertIsNotNone(env.grid)
        self.assertIsNotNone(env.goal)

    def test_move_agent(self):
        name = "Test Maze Environment"
        description = "Test description"
        width = 4
        height = 4
        env = MazeEnvironment(name, description, width,
                              height, goal_x=3, goal_y=3, start_x=0, start_y=0, maze_path='test_maze.txt')
        # Start at position (0, 0)
        # Test invalid move
        self.assertRaises(ValueError, env._move_agent, "INVALID")
        self.assertRaises(ValueError, env._move_agent, "DOWN")
        # Test moves, they assert false because it is not game over
        self.assertFalse(env._move_agent(direction=Direction.UP)[0])
        self.assertFalse(env._move_agent(direction=Direction.DOWN)[0])
        self.assertFalse(env._move_agent(direction=Direction.RIGHT)[0])

    def test_reset(self):
        name = "Test Maze Environment"
        description = "Test description"
        width = 4
        height = 4
        env = MazeEnvironment(name, description, width,
                              height, goal_x=3, goal_y=3, start_x=0, start_y=0, maze_path='test_maze.txt')
        env.reset()
        self.assertIsNotNone(env.agent)
        self.assertIsNotNone(env.grid)
        self.assertEqual(env.agent.pos_x, 0)
        self.assertEqual(env.agent.pos_y, 0)

    def test_step(self):
        name = "Test Maze Environment"
        description = "Test description"
        width = 4
        height = 4
        env = MazeEnvironment(name, description, width,
                              height, goal_x=3, goal_y=3, start_x=0, start_y=0, maze_path='test_maze.txt')
        env.step("DOWN")
        self.assertEqual(env.agent.pos_x, 0)
        self.assertEqual(env.agent.pos_y, 1)

        env.step("RIGHT")
        self.assertEqual(env.agent.pos_x, 0)
        self.assertEqual(env.agent.pos_y, 1)


if __name__ == '__main__':
    unittest.main()
