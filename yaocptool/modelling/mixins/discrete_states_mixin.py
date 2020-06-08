import collections
from contextlib import suppress
from itertools import islice

from casadi.casadi import DM, SX, substitute, vec, vertcat

from yaocptool.modelling.utils import Derivative, EqualityEquation, NextK
from yaocptool.util.util import remove_variables_from_vector


class DiscreteStateMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x = SX([])
        self.x_0 = SX([])
        self._diff = dict()

    @property
    def n_x(self):
        return self.x.numel()

    @property
    def x_names(self):
        return [self.x[i].name() for i in range(self.n_x)]

    @property
    def diff(self):
        try:
            return vertcat(
                *[val for val in self._diff.values() if val is not None])
        except NotImplementedError:
            return SX.zeros(0, 1)

    def create_state(self, name="x", size=1):
        """
        Create a new state with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new state will be vectorized (casadi.vec) to be
        included in the state vector (model.x).

        :param name: str
        :param size: int|tuple
        :return:
        """
        if callable(getattr(self, 'name_variable', None)):
            name = self.name_variable(name)

        new_x = SX.sym(name, size)
        new_x_0 = SX.sym(name + "_0_sym", size)
        self.include_state(vec(new_x), diff=None, x_0=vec(new_x_0))
        return new_x

    def include_state(self, var, diff=None, x_0=None):
        n_x = var.numel()
        self.x = vertcat(self.x, var)

        if x_0 is None:
            x_0 = vertcat(*[SX.sym(var_i.name()) for var_i in var.nz])
        self.x_0 = vertcat(self.x_0, x_0)

        # crate entry for included state
        for ind, x_i in enumerate(var.nz):
            if x_i in self._diff:
                raise ValueError(f'State "{x_i}" already in this model')
            self._diff[x_i] = None
        if diff is not None:
            self.include_equations(diff=diff, x=var)
        return x_0

    def remove_state(self, var, eq=None):
        self.x = remove_variables_from_vector(var, self.x)

        for x_i in var.nz:
            del self._diff[x_i]

    def replace_variable(self, original, replacement):
        if isinstance(original, list):
            original = vertcat(*original)
        if isinstance(replacement, list):
            replacement = vertcat(*replacement)

        if not original.numel() == replacement.numel():
            raise ValueError(
                "Original and replacement must have the same number of elements!"
                "original.numel()={}, replacement.numel()={}".format(
                    original.numel(), replacement.numel()))

        if callable(getattr(super(), 'replace_variable', None)):
            super().replace_variable(original, replacement)

        if original.numel() > 0:
            for x_i, x_i_eq in self._diff.items():
                self._diff[x_i] = substitute(x_i_eq, original, replacement)

    def include_equations(self, *args, **kwargs):
        if callable(getattr(super(), 'include_equations', None)):
            super().include_equations(*args, **kwargs)

        diff = kwargs.pop('diff', None)
        x = kwargs.pop('x', None)
        if diff is None and x is not None:
            raise ValueError("`diff` is None but `x` is not None")

        # if is in the list form
        if isinstance(diff, collections.abc.Sequence):
            diff = vertcat(*diff)

        if isinstance(x, collections.abc.Sequence):
            x = vertcat(*x)

        # if diff was passed but not x, try to guess the x
        if x is None and diff is not None:
            # Check if None are all sequential, ortherwise we don't know who it belongs
            first_none = list(self._diff.values()).index(None)
            if not all(eq is None
                       for eq in islice(self._diff.values(), 0, first_none)):
                raise ValueError(
                    "diff should be inserted on the equation form or in the list form."
                    "You can't mix both without explicit passing the states associated with the equation."
                )
            x = vertcat(*list(self._diff.keys())[first_none:first_none +
                                                 diff.numel()])

        if len(args) > 0 and diff is None:
            x = SX([])
            diff = SX([])

        # get diff and x from equality equations
        for eq in args:
            if isinstance(eq, EqualityEquation):
                if isinstance(eq.lhs, NextK):
                    diff = vertcat(diff, eq.rhs)
                    x = vertcat(x, eq.lhs.inner)

        # actually include the equations
        if diff is not None and diff.numel() > 0:
            for x_i in vertcat(x).nz:
                if self._diff[x_i] is not None:
                    raise Warning(
                        f'State "{x_i}" already had an diff associated, overriding it!'
                    )
            diff_dict = dict(self._diff)
            diff_dict.update({x_i: diff[ind] for ind, x_i in enumerate(x.nz)})
            self._diff = diff_dict
