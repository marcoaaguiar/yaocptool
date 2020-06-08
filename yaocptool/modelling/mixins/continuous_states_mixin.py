import collections
from contextlib import suppress
from itertools import islice

from casadi.casadi import DM, SX, substitute, vec, vertcat

from yaocptool.modelling.utils import Derivative, EqualityEquation
from yaocptool.util.util import remove_variables_from_vector


class ContinuousStateMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x = SX([])
        self.x_0 = SX([])
        self._ode = dict()

    @property
    def n_x(self):
        return self.x.numel()

    @property
    def x_names(self):
        return [self.x[i].name() for i in range(self.n_x)]

    @property
    def ode(self):
        try:
            return vertcat(
                *[val for val in self._ode.values() if val is not None])
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
        self.include_state(vec(new_x), ode=None, x_0=vec(new_x_0))
        return new_x

    def include_state(self, var, ode=None, x_0=None):
        n_x = var.numel()
        self.x = vertcat(self.x, var)

        if x_0 is None:
            x_0 = vertcat(*[SX.sym(var_i.name()) for var_i in var.nz])
        self.x_0 = vertcat(self.x_0, x_0)

        # crate entry for included state
        for ind, x_i in enumerate(var.nz):
            if x_i in self._ode:
                raise ValueError(f'State "{x_i}" already in this model')
            self._ode[x_i] = None
        if ode is not None:
            self.include_equations(ode=ode, x=var)
        return x_0

    def remove_state(self, var, eq=None):
        self.x = remove_variables_from_vector(var, self.x)

        for x_i in var.nz:
            del self._ode[x_i]

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
            for x_i, x_i_eq in self._ode.items():
                self._ode[x_i] = substitute(x_i_eq, original, replacement)

    def include_equations(self, *args, **kwargs):
        if callable(getattr(super(), 'include_equations', None)):
            super().include_equations(*args, **kwargs)

        ode = kwargs.pop('ode', None)
        x = kwargs.pop('x', None)
        if ode is None and x is not None:
            raise ValueError("`ode` is None but `x` is not None")

        # if is in the list form
        if isinstance(ode, collections.abc.Sequence):
            ode = vertcat(*ode)

        if isinstance(x, collections.abc.Sequence):
            x = vertcat(*x)

        # if ode was passed but not x, try to guess the x
        if x is None and ode is not None:
            # Check if None are all sequential, ortherwise we don't know who it belongs
            first_none = list(self._ode.values()).index(None)
            if not all(eq is None
                       for eq in islice(self._ode.values(), 0, first_none)):
                raise ValueError(
                    "ODE should be inserted on the equation form or in the list form."
                    "You can't mix both without explicit passing the states associated with the equation."
                )
            x = vertcat(*list(self._ode.keys())[first_none:first_none +
                                                ode.numel()])

        if len(args) > 0 and ode is None:
            x = SX([])
            ode = SX([])

        # get ode and x from equality equations
        for eq in args:
            if isinstance(eq, EqualityEquation):
                if isinstance(eq.lhs, Derivative):
                    ode = vertcat(ode, eq.rhs)
                    x = vertcat(x, eq.lhs.inner)

        # actually include the equations
        if ode is not None and ode.numel() > 0:
            for x_i in vertcat(x).nz:
                if self._ode[x_i] is not None:
                    raise Warning(
                        f'State "{x_i}" already had an ODE associated, overriding it!'
                    )
            ode_dict = dict(self._ode)
            ode_dict.update({x_i: ode[ind] for ind, x_i in enumerate(x.nz)})
            self._ode = ode_dict
