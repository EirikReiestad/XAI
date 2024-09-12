from demo import settings, DemoType, RLType
from demo.src.demos.single_agent.maze import MazeDemo as Demo

"""
match settings.DEMO_TYPE:
    case DemoType.CARTPOLE:
        from demo.src.demos.single_agent.cartpole import CartPoleDemo as Demo
    case DemoType.MAZE:
        from demo.src.demos.single_agent.maze import MazeDemo as Demo
    case DemoType.COOP:
        from demo.src.demos.multi_agent.dqn.coop import CoopDemo as Demo
    case DemoType.TAG:
        if settings.RL_TYPE == RLType.DQN:
            from demo.src.demos.multi_agent.dqn.tag import TagDemo as Demo
        elif settings.RL_TYPE == RLType.PPO:
            from demo.src.demos.multi_agent.ppo.tag import TagDemo as Demo
        else:
            raise ValueError("Invalid RL type")
    case _:
        raise ValueError("Invalid demo type")
"""

demo = Demo()
demo.run()
