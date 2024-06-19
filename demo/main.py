from .src.snake_dqn import main as snake_main
from .src.maze_qlearning import main as maze_qlearning_main
from .src.maze_dqn import main as maze_dqn_main
from .src.cartpole_dqn import main as cartpole_main

if __name__ == "__main__":
    # snake_main()
    # maze_dqn_main()
    cartpole_main()
