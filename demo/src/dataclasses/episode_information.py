from dataclasses import dataclass

@dataclass
class EpisodeInformation:
    duration: list[int]
    rewards: list[float]
