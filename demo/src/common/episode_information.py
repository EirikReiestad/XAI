from dataclasses import dataclass

@dataclass
class EpisodeInformation:
    durations: list[int]
    rewards: list[float]
