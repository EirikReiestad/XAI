import enum
import logging
from dataclasses import dataclass


class BootcampName(enum.Enum):
    HIDER = 0
    SEEKER = 1
    SLOW_HIDER_10 = 2
    SLOW_HIDER_5 = 3
    SLOW_HIDER_2 = 4
    COMBINED = 5


@dataclass
class BootcampTrainingSteps:
    hider = 10000
    seeker = 20000
    slow_hider_10 = 50000
    slow_hider_5 = 50000
    slow_hider_2 = 100000

    def get_days(self, name: BootcampName):
        if name == BootcampName.HIDER:
            return self.hider
        if name == BootcampName.SEEKER:
            return self.seeker
        if name == BootcampName.SLOW_HIDER_10:
            return self.slow_hider_10
        if name == BootcampName.SLOW_HIDER_5:
            return self.slow_hider_5
        if name == BootcampName.SLOW_HIDER_2:
            return self.slow_hider_2
        return 0


class Bootcamp:
    def __init__(self):
        self._name = BootcampName.HIDER
        self._training_days = 0
        logging.info(f"Starting {self._name} bootcamp")

    def train(self):
        self._training_days += 1
        self._next()

    def move_hider(self, steps: int) -> bool:
        if self._name == BootcampName.HIDER:
            return True
        if self._name == BootcampName.SLOW_HIDER_10 and steps % 10 == 0:
            return True
        if self._name == BootcampName.SLOW_HIDER_5 and steps % 5 == 0:
            return True
        if self._name == BootcampName.SLOW_HIDER_2 and steps % 2 == 0:
            return True
        if self._name == BootcampName.COMBINED:
            return True
        return False

    def _next(self):
        if self._training_days < BootcampTrainingSteps().get_days(self._name):
            return
        if self._name == BootcampName.COMBINED:
            return
        self._training_days = 0
        old_name = self._name
        self._name = BootcampName(self._name.value + 1)
        logging.info(f"{old_name} bootcamp completed, starting {self._name} bootcamp")

    @property
    def name(self):
        return self._name
