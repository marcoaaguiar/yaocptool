import abc
from typing import List, Union
from casadi import SX


class BaseMixin(abc.ABC):
    x: SX
    y: SX
    u: SX
    p: SX
    theta: SX
    t: SX
    tau: SX

    @abc.abstractmethod
    def name_variable(self, name: str) -> str:
        pass

    @abc.abstractmethod
    def include_equations(self, *args: SX, **kwargs: Union[SX, List[SX]]):
        pass

    @abc.abstractmethod
    def replace_variable(self, original: SX, replacement: SX):
        pass
