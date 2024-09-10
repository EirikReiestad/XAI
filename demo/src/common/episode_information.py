from dataclasses import dataclass


@dataclass
class EpisodeInformation:
    durations: list[int]
    rewards: list[float]
    object_moved_distance: list[float]

    def last_episode(self, prefix: str = ""):
        return {
            f"{prefix}durations": self.durations[-1],
            f"{prefix}rewards": self.rewards[-1],
            f"{prefix}object_moved_distance": self.object_moved_distance[-1],
        }
