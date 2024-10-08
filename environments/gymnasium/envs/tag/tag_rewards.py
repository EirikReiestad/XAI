import math

from environments.gymnasium.envs.tag import rewards
from environments.gymnasium.utils import Position


class TagRewards:
    max_distance = 1

    def __init__(self):
        self.rewards = {
            "tagged": rewards.TAGGED_REWARD,
            "not_tagged": rewards.NOT_TAGGED_REWARD,
            "move": rewards.MOVE_REWARD,
            "end": rewards.END_REWARD,
            "terminated": rewards.TERMINATED_REWARD,
            "truncated": rewards.TRUNCATED_REWARD,
            "collision": rewards.COLLISION_REWARD,
            "wrong_grab": rewards.WRONG_GRAB_RELEASE_REWARD,
        }

    def reset(self):
        self.last_distance = 0

    def get_tag_reward(
        self,
        agent: Position,
        other_agent: Position,
        terminated: bool,
        radius: float = 1,
    ) -> tuple[tuple[float, float], bool]:
        distance = agent.distance_to(other_agent)
        self.max_distance = max(self.max_distance, distance)
        normalized_distance = distance / self.max_distance
        exp_distance = 1 - math.exp(-normalized_distance)

        if distance <= radius or terminated:
            tagged_reward = (
                self.tagged_reward[0] + (1 - exp_distance),
                self.tagged_reward[1] + exp_distance,
            )
            return tagged_reward, True

        not_tagged_reward = (
            self.not_tagged_reward[0] + (1 - exp_distance),
            self.not_tagged_reward[1] + exp_distance,
        )
        return not_tagged_reward, False

    @property
    def config(self):
        return {
            "tagged": self.tagged_reward,
            "not_tagged": self.not_tagged_reward,
            "move": self.move_reward,
            "end": self.end_reward,
            "terminated": self.terminated_reward,
            "truncated": self.truncated_reward,
            "collision": self.collision_reward,
            "wrong_grab": self.wrong_grab_release_reward,
        }

    @property
    def tagged_reward(self):
        return self.rewards["tagged"]

    @property
    def not_tagged_reward(self):
        return self.rewards["not_tagged"]

    @property
    def move_reward(self):
        return self.rewards["move"]

    @property
    def collision_reward(self):
        return self.rewards["collision"]

    @property
    def end_reward(self):
        return self.rewards["end"]

    @property
    def terminated_reward(self):
        return self.rewards["terminated"]

    @property
    def truncated_reward(self):
        return self.rewards["truncated"]

    @property
    def wrong_grab_release_reward(self):
        return self.rewards["wrong_grab"]
