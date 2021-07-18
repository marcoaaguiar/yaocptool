import signal
from types import TracebackType
from typing import Type


class timeout:
    def __init__(
        self, seconds: int = 1, raise_exc: bool = True, error_message: str = "Timeout"
    ):
        self.seconds = seconds
        self.raise_exc = raise_exc
        self.error_message = error_message
        self._timedout = None

    @property
    def timedout(self):
        #  if self._timedout is None:
        #      raise ValueError("timeout hasn't finished yet")
        return self._timedout

    def handle_timeout(self, signum, frame):
        self._timedout = True
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException],
        value: BaseException,
        traceback: TracebackType,
    ):
        signal.alarm(0)
        if exc_type is None:
            return True

        max_depth = 20
        cause = value
        for _ in range(max_depth):
            if cause.__cause__ is None:
                break
            cause = cause.__cause__

        if type(cause) is not TimeoutError:
            self._timedout = False
            return

        self._timedout = True
        return not self.raise_exc
