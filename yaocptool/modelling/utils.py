from casadi import SX


class EqualityEquation:
    def __init__(self, lhs: "Derivative", rhs: SX):
        self.rhs = rhs
        self.lhs = lhs

    def __str__(self) -> str:
        return f"EqualityEquation({self.lhs} == {self.rhs})"


class Derivative:
    def __init__(self, inner: SX):
        self.inner = inner

    def __eq__(self, other: object) -> EqualityEquation:  # type: ignore
        if not isinstance(other, SX):
            raise NotImplemented
        return EqualityEquation(self, other)

    def __str__(self) -> str:
        return f"der({self.inner})"


def der(inner: SX) -> Derivative:
    return Derivative(inner)
