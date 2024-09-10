from dataclasses import dataclass


@dataclass
class EpisodeInformation:
    durations: list[int]
    rewards: list[float]
    object_moved_distance: list[float]

    def last_episode(self, prefix: str = ""):
        last_episode = {}
        for key in ["durations", "rewards", "object_moved_distance"]:
            if getattr(self, key):
                last_episode[f"{prefix}{key}"] = getattr(self, key)[-1]
        return last_episode
