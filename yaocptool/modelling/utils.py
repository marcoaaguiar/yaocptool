class EqualityEquation:
    def __init__(self, lhs, rhs):
        self.rhs = rhs
        self.lhs = lhs

    def __str__(self):
        return f"EqualityEquation({self.lhs} == {self.rhs})"


class Derivative:
    def __init__(self, inner):
        self.inner = inner

    def __eq__(self, other):
        return EqualityEquation(self, other)

    def __str__(self):
        return f"der({self.inner})"


class NextK:
    def __init__(self, inner):
        self.inner = inner

    def __eq__(self, other):
        return EqualityEquation(self, other)

    def __str__(self):
        return f"next_k({self.inner})"


def der(*args, **kwargs):
    return Derivative(*args, **kwargs)


def next_k(*args, **kwargs):
    return NextK(*args, **kwargs)
