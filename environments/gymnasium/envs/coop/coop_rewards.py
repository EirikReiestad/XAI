from environments import settings
from environments.gymnasium.utils import Position


class CoopRewards:
    def __init__(self):
        self.rewards = {
            "goal": settings.GOAL_REWARD,
            "move": settings.MOVE_REWARD,
            "terminated": settings.TERMINATED_REWARD,
            "truncated": settings.TRUNCATED_REWARD,
        }

    def get_reward(self, agent: Position, goal: Position, collided: bool):
        if agent == goal:
            return self.rewards["goal"]
        elif collided:
            return self.rewards["terminated"]
        else:
            return self.rewards["move"]

    @property
    def goal_reward(self):
        return self.rewards["goal"]

    @property
    def move_reward(self):
        return self.rewards["move"]

    @property
    def terminated_reward(self):
        return self.rewards["terminated"]

    @property
    def truncated_reward(self):
        return self.rewards["truncated"]
