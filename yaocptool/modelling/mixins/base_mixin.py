import abc
from casadi import SX


class BaseMixin(abc.ABCMeta):
    t: SX

    @abc.abstractmethod
    def name_variable(self, name: str) -> str:
        return

    @abc.abstractmethod
    def include_equations(self, *args: SX, **kwargs: SX):
        return

    @abc.abstractmethod
    def replace_variable(self, original: SX, replacement: SX):
        return
