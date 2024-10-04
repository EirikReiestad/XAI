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

    def get_tag_reward(
        self,
        agent: Position,
        other_agent: Position,
        terminated: bool,
        radius: float = 1,
    ) -> tuple[tuple[float, float], bool]:
        distance = agent.distance_to(other_agent)
        if distance <= radius or terminated:
            return self.tagged_reward, True
        self.max_distance = max(self.max_distance, distance)
        max_distance = self.max_distance * 100
        distance_rewards = (
            1 - distance / max_distance,
            distance / max_distance,
        )
        not_tagged_reward = tuple(
            not_tagged + dist
            for not_tagged, dist in zip(distance_rewards, self.not_tagged_reward)
        )
        return not_tagged_reward, False

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
