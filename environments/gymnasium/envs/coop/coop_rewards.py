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

    def get_individual_reward(self, collided: bool):
        if collided:
            return self.terminated_reward
        else:
            return self.move_reward

    def get_cooperative_reward(
        self, agent: Position, other_agent: Position, radius: float = 2
    ) -> tuple[float, bool]:
        if agent.distance_to(other_agent) < radius:
            return self.goal_reward, True
        return 0, False

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
