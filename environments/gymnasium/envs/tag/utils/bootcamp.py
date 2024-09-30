import enum
import logging
from dataclasses import dataclass


class BootcampName(enum.Enum):
    HIDER = 0
    SEEKER = 1
    COMBINED = 2


@dataclass
class BootcampTrainingSteps:
    hider = 1000
    seeker = 10000

    def get_days(self, name: BootcampName):
        if name == BootcampName.HIDER:
            return self.hider
        if name == BootcampName.SEEKER:
            return self.seeker
        return 0


class Bootcamp:
    def __init__(self):
        self._name = BootcampName.HIDER
        self._training_days = 0

    def train(self):
        self._training_days += 1
        self._next()

    def _next(self):
        if self._training_days < BootcampTrainingSteps().get_days(self._name):
            return
        if self._name == BootcampName.COMBINED:
            return
        old_name = self._name
        self._name = BootcampName((self._name.value + 1) % 3)
        logging.info(f"{old_name} bootcamp completed, starting {self._name} bootcamp")

    @property
    def name(self):
        return self._name
