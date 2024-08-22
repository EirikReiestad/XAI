import enum
import logging


@enum.unique
class StateType(enum.Enum):
    FULL = "full"
    PARTIAL = "partial"
    RGB = "rgb"

    def __str__(self):
        return self.value

    @staticmethod
    def from_string(state: str) -> "StateType":
        """
        Convert a string to a StateType enum value.

        Args:
            state (str): The state type as a string.

        Returns:
            StateType: Corresponding StateType enum value.

        Raises:
            ValueError: If the string does not match any StateType.
        """
        try:
            return StateType(state)
        except ValueError:
            logging.error(f"Invalid state type: {state}")
            raise ValueError(f"Invalid state type: {state}")
