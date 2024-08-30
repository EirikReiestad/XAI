from demo import settings, DemoType

match settings.DEMO:
    case DemoType.MAZE:
        from demo.src.demos.single_agent.maze import MazeDemo as Demo
    case DemoType.COOP:
        from demo.src.demos.multi_agent.coop import CoopDemo as Demo
    case DemoType.TAG:
        from demo.src.demos.multi_agent.tag import TagDemo as Demo
    case _:
        raise ValueError("Invalid demo type")

demo = Demo()
demo.run()
