"""Created on Mar 25 15:10:29 2025"""


class BaseFitterError(Exception):
    pass


class BoundaryInconsistentWithGuess(BaseFitterError):
    def __init__(self, message):
        # Pass the full message to the base Exception class
        full_message = "Number of parameters in p0 is greater than the model requires. " + message
        super().__init__(full_message)  # Correct way to store the message
