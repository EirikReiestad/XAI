from environments.gymnasium.utils import Position
from environments.gymnasium.envs.coop import rewards


class CoopRewards:
    def __init__(self):
        self.rewards = {
            "goal": rewards.GOAL_REWARD,
            "move": rewards.MOVE_REWARD,
            "terminated": rewards.TERMINATED_REWARD,
            "truncated": rewards.TRUNCATED_REWARD,
        }

    def get_individual_reward(self, collided: bool):
        if collided:
            return self.terminated_reward
        else:
            return self.move_reward

    def get_cooperative_reward(
        self, agent: Position, other_agent: Position, radius: float = 1
    ) -> tuple[tuple[float, float], bool]:
        if agent.distance_to(other_agent) <= radius:
            return (self.goal_reward, self.goal_reward), True
        return (0, 0), False

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
