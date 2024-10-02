import enum
import logging
from dataclasses import dataclass


class BootcampName(enum.Enum):
    HIDER = 0
    SEEKER = 1
    SLOW_HIDER = 2
    COMBINED = 3


@dataclass
class BootcampTrainingSteps:
    hider = 10000
    seeker = 10000
    slow_hider = 30000

    def get_days(self, name: BootcampName):
        if name == BootcampName.HIDER:
            return self.hider
        if name == BootcampName.SEEKER:
            return self.seeker
        if name == BootcampName.SLOW_HIDER:
            return self.slow_hider
        return 0


class Bootcamp:
    def __init__(self):
        self._name = BootcampName.HIDER
        self._training_days = 0
        self.slow_hider_factor = 10
        self.slow_hider_step_factor = 1

        self.__post_init__()

    def __post_init__(self):
        logging.info(f"Starting {self._name} bootcamp")

    def train(self):
        self._training_days += 1
        self._next()

    def move_hider(self, steps: int) -> bool:
        if self._name == BootcampName.HIDER:
            return True
        if (
            self._name == BootcampName.SLOW_HIDER
            and steps % self.slow_hider_factor == 0
        ):
            return True
        if self._name == BootcampName.COMBINED:
            return True
        return False

    def _next(self):
        if self._training_days < BootcampTrainingSteps().get_days(self._name):
            return
        if self._name == BootcampName.COMBINED:
            return
        if self._name == BootcampName.SLOW_HIDER:
            self.slow_hider_factor -= self.slow_hider_step_factor
            self._training_days = 0
            if self.slow_hider_factor > 1:
                logging.info(
                    f"Continuing with bootcamp {self._name} with slow hider factor: {self.slow_hider_factor}"
                )
                return
        self._training_days = 0
        old_name = self._name
        self._name = BootcampName(self._name.value + 1)
        logging.info(f"{old_name} bootcamp completed, starting {self._name} bootcamp")

    @property
    def name(self):
        return self._name
