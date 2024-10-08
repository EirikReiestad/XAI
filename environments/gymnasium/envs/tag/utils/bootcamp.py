import enum
import logging
from dataclasses import dataclass


class BootcampName(enum.Enum):
    HIDER = 0
    SEEKER = 1
    SLOW_AGENT = 2
    COMBINED = 3


@dataclass
class BootcampTrainingSteps:
    hider = 1
    seeker = 1
    slow_agent = 1000

    def get_days(self, name: BootcampName):
        if name == BootcampName.HIDER:
            return self.hider
        if name == BootcampName.SEEKER:
            return self.seeker
        if name == BootcampName.SLOW_AGENT:
            return self.slow_agent
        return 0


class Bootcamp:
    def __init__(self):
        self._name = BootcampName.HIDER
        self._training_days = 0
        self.slow_factor = 20
        self.slow_step_factor = 1
        self.slow_agent = 0

    def step(self):
        self._training_days += 1
        self._next()

    def move_hider(self, steps: int) -> bool:
        if self.slow_agent == 1:
            return True
        if self._name == BootcampName.HIDER:
            return True
        if self._name == BootcampName.SLOW_AGENT and steps % self.slow_factor == 0:
            return True
        if self._name == BootcampName.COMBINED:
            return True
        return False

    def move_seeker(self, steps: int) -> bool:
        if self.slow_agent == 0:
            return True
        if self._name == BootcampName.SEEKER:
            return True
        if self._name == BootcampName.SLOW_AGENT and steps % self.slow_factor == 0:
            return True
        if self._name == BootcampName.COMBINED:
            return True
        return False

    def _next(self):
        if self._training_days < BootcampTrainingSteps().get_days(self._name):
            return
        if self._name == BootcampName.COMBINED:
            return
        if self._name == BootcampName.SLOW_AGENT:
            self._training_days = 0
            self.slow_agent = 1 if self.slow_agent == 0 else 0
            if self.slow_agent == 1:
                self.slow_factor -= self.slow_step_factor
            if self.slow_factor > 1:
                logging.info(
                    f"Continuing with bootcamp {self._name} with slow factor: {self.slow_factor} for agent {self.slow_agent}"
                )
                return
        self._training_days = 0
        old_name = self._name
        self._name = BootcampName(self._name.value + 1)
        logging.info(f"{old_name} bootcamp completed, starting {self._name} bootcamp")

    @property
    def name(self):
        return self._name
