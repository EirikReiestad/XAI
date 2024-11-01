import math

import numpy as np

from environments.gymnasium.envs.tag import rewards
from environments.gymnasium.utils import Position


class TagRewards:
    max_distance = 1
    last_distance = np.inf

    tagged_reward: tuple[float, float] = rewards.TAGGED_REWARD
    not_tagged_reward: tuple[float, float] = rewards.NOT_TAGGED_REWARD
    can_see_reward: tuple[float, float] = rewards.CAN_SEE_REWARD
    move_reward: tuple[float, float] = rewards.MOVE_REWARD
    end_reward: tuple[float, float] = rewards.END_REWARD
    terminated_reward: float = rewards.TERMINATED_REWARD
    truncated_reward: tuple[float, float] = rewards.TRUNCATED_REWARD
    collision_reward: float = rewards.COLLISION_REWARD
    wrong_grab_release_reward: float = rewards.WRONG_GRAB_RELEASE_REWARD
    distance_factor: float = rewards.DISTANCE_REWARD_FACTOR
    move_towards_reward: tuple[float, float] = rewards.MOVE_TOWARDS_REWARD
    move_away_reward: tuple[float, float] = rewards.MOVE_AWAY_REWARD

    def reset(self):
        self.max_distance = 1
        self.last_distance = np.inf

    def get_tag_reward(
        self,
        agent: Position,
        other_agent: Position,
        has_direct_sight: bool,
        terminated: bool,
        radius: float = 1,
    ) -> tuple[tuple[float, float], bool]:
        distance_reward, tagged = self._get_distance_reward(
            agent, other_agent, terminated, radius
        )
        can_see_reward = (0, 0)
        if has_direct_sight:
            can_see_reward = self.can_see_reward

        reward = (
            distance_reward[0] + can_see_reward[0],
            distance_reward[1] + can_see_reward[1],
        )
        return reward, tagged

    def _get_distance_reward(
        self, agent: Position, other_agent: Position, terminated: bool, radius: float
    ) -> tuple[tuple[float, float], bool]:
        distance = agent.distance_to(other_agent)
        self.max_distance = max(self.max_distance, distance)
        normalized_distance = distance / self.max_distance
        exp_distance = math.exp(-normalized_distance)

        distance_reward = (1 - normalized_distance) - 0.5

        seeker_distance_reward = distance_reward * self.distance_factor
        hider_distance_reward = -distance_reward * self.distance_factor

        if distance < self.last_distance:
            self.last_distance = distance
            seeker_distance_reward += self.move_towards_reward[0]
            hider_distance_reward += self.move_towards_reward[1]
        elif distance > self.last_distance:
            self.last_distance = distance
            seeker_distance_reward += self.move_away_reward[0]
            hider_distance_reward += self.move_away_reward[1]

        if distance <= radius or terminated:
            tagged_reward = (
                self.tagged_reward[0] + seeker_distance_reward,
                self.tagged_reward[1] + hider_distance_reward,
            )
            return tagged_reward, True
        not_tagged_reward = (
            self.not_tagged_reward[0] + seeker_distance_reward,
            self.not_tagged_reward[1] + hider_distance_reward,
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
            "distance_factor": self.distance_factor,
            "move_towards": self.move_towards_reward,
            "move_away": self.move_away_reward,
        }
