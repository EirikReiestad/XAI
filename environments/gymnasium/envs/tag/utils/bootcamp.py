import enum
import logging
from dataclasses import dataclass
from .agent_type import AgentType


class BootcampName(enum.Enum):
    HIDER = 0
    SEEKER = 1
    SLOW_AGENT = 2
    COMBINED = 3
    FINISHED = 4


@dataclass
class BootcampTrainingSteps:
    hider = 500
    seeker = 500
    slow_agent = 1000
    combined = 1000

    def get_days(self, name: BootcampName):
        if name == BootcampName.HIDER:
            return self.hider
        if name == BootcampName.SEEKER:
            return self.seeker
        if name == BootcampName.SLOW_AGENT:
            return self.slow_agent
        if name == BootcampName.COMBINED:
            return self.combined
        return 0


class Bootcamp:
    def __init__(self):
        self._name = BootcampName.HIDER
        self._training_days = 0
        self._bootcamp_num = 0
        self._num_bootcamps = 1

        self.initial_slow_factor = 20
        self.slow_factors: list[int] = self._get_slow_factors(
            self.initial_slow_factor, self._num_bootcamps
        )
        self._slow_factor = 0
        self.slow_step_factor = 1
        self.slow_agent = 0

        self.slow_hider_factor = 4
        self.slow_seeker_factor = 1

    def step(self):
        self._training_days += 1
        self._next()

    def move_hider(self, steps: int) -> bool:
        if self.name == BootcampName.HIDER:
            return True
        if self.name == BootcampName.SLOW_AGENT and steps % self.slow_factor == 0:
            return True
        if self.name == BootcampName.COMBINED and steps % self.slow_hider_factor == 0:
            return True
        if self.name == BootcampName.FINISHED and steps % self.slow_seeker_factor == 0:
            return True
        if self.slow_agent == 1:
            return True
        return False

    def move_seeker(self, steps: int) -> bool:
        if self.name == BootcampName.SEEKER:
            return True
        if self.name == BootcampName.SLOW_AGENT and steps % self.slow_factor == 0:
            return True
        if self.name == BootcampName.COMBINED and steps % self.slow_seeker_factor == 0:
            return True
        if self.name == BootcampName.FINISHED and steps % self.slow_seeker_factor == 0:
            return True
        if self.slow_agent == 0:
            return True
        return False

    def agent_slow_factor(self, agent: AgentType) -> int:
        if self.name == BootcampName:
            if agent == AgentType.SEEKER:
                return self.slow_seeker_factor
            if agent == AgentType.HIDER:
                return self.slow_hider_factor
        if agent == AgentType.SEEKER:
            return max(self.slow_factor, self.slow_seeker_factor)
        if agent == AgentType.HIDER:
            return max(self.slow_factor, self.slow_hider_factor)
        raise ValueError(f"Unknown agent type: {agent}")

    def _next(self):
        bootcamp_days = BootcampTrainingSteps().get_days(self._name)
        if self._training_days < bootcamp_days:
            return
        if self._name == BootcampName.FINISHED:
            if self._bootcamp_num >= self._num_bootcamps - 1:
                return
            old_name = self._name
            self._reset_bootcamp()
            logging.info(
                f"Starting Bootcamp #{self._bootcamp_num}, starting Bootcamp #{self._bootcamp_num} {self._name}"
            )
            return
        if self._name == BootcampName.SLOW_AGENT:
            self._training_days = 0
            self.slow_agent = 1 if self.slow_agent == 0 else 0
            if self.slow_agent == 1:
                self.slow_factor -= self.slow_step_factor
            if self.slow_factor > 1:
                logging.info(
                    f"Continuing with Bootcamp #{self._bootcamp_num} {self._name} with slow factor: {self.slow_factor} for agent {self.slow_agent}"
                )
                return
        self._training_days = 0
        old_name = self._name
        self._name = BootcampName(self._name.value + 1)
        if self._name == BootcampName.FINISHED:
            self._next()
        bootcamp_days = BootcampTrainingSteps().get_days(self._name)
        logging.info(
            f"{old_name} Bootcamp completed, starting {self._name} Bootcamp #{self._bootcamp_num} for {bootcamp_days} days"
        )

    def _reset_bootcamp(self):
        self._bootcamp_num += 1
        self._name = BootcampName.HIDER
        self._training_days = 0

    def _get_slow_factors(self, n: int, parts: int) -> list[int]:
        if parts == 1:
            return [10]
        parts -= 1
        avergage_slow_factor = n // parts
        result = [n]
        for i in range(1, parts):
            part = result[i - 1] - avergage_slow_factor
            assert (
                part > 1
            ), f"Invalid slow factor: {part}. It should be greater than 1."
            result.append(part)

        result = [int(x) for x in result]
        result.append(1)

        assert (
            len(result) == parts + 1
        ), f"Invalid slow factors: {result}. It should have {parts} elements."
        return result

    @property
    def slow_factor(self):
        return self.slow_factors[self._bootcamp_num]

    @slow_factor.setter
    def slow_factor(self, value: int):
        self.slow_factors[self._bootcamp_num] = value

    @property
    def name(self):
        return self._name
