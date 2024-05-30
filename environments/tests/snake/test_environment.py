import unittest

from src.snake.environment import SnakeEnvironment
from src.snake.direction import Direction


class TestSnakeEnvironment(unittest.TestCase):
    def test_init(self):
        name = "Test Snake Environment"
        description = "Test description"
        width = 10
        height = 10
        env = SnakeEnvironment(name, description, width, height)
        self.assertEqual(env.name, name)
        self.assertEqual(env.description, description)
        self.assertEqual(env.grid.width, width)
        self.assertEqual(env.grid.height, height)
        self.assertIsNotNone(env.snake)
        self.assertIsNotNone(env.food)

    def test_move_snake(self):
        name = "Test Snake Environment"
        description = "Test description"
        width = 10
        height = 10
        env = SnakeEnvironment(name, description, width, height)
        # Test invalid move
        self.assertRaises(AssertionError, env.move_snake, "INVALID")
        self.assertRaises(AssertionError, env.move_snake, "DOWN")
        # This is not a valid move since the snake is moving right
        self.assertTrue(env.move_snake(direction=Direction.LEFT))
        # Test valid move
        self.assertTrue(env.move_snake(direction=Direction.UP))
        self.assertTrue(env.move_snake(direction=Direction.DOWN))
        self.assertTrue(env.move_snake(direction=Direction.RIGHT))

    def test_grow_snake(self):
        name = "Test Snake Environment"
        description = "Test description"
        width = 10
        height = 10
        env = SnakeEnvironment(name, description, width, height)
        # Test grow snake
        # Initial length is 3
        self.assertEqual(len(env.snake), 3)

        env.snake.grow()
        env.move_snake(direction=Direction.RIGHT)
        self.assertEqual(len(env.snake), 4)


if __name__ == '__main__':
    unittest.main()
