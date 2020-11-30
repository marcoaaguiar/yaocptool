from collections import abc
from itertools import islice
from typing import Dict, List, Optional, Tuple, Union, overload
import warnings
from yaocptool.modelling.mixins.base_mixin import BaseMixin

from casadi import DM, SX, substitute, vec, vertcat

from yaocptool.modelling.utils import Derivative, EqualityEquation
from yaocptool.util.util import remove_variables_from_vector


class ContinuousStateMixin(BaseMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.x = SX(0, 1)
        self.x_0 = SX(0, 1)
        self._ode: Dict[SX, Optional[SX]] = {}

    @property
    def n_x(self) -> int:
        return self.x.numel()

    @property
    def x_names(self) -> List[str]:
        return [self.x[i].name() for i in range(self.n_x)]

    @property
    def ode(self) -> SX:
        try:
            return SX(vertcat(*[val for val in self._ode.values() if val is not None]))
        except NotImplementedError:
            return SX()

    def create_state(
        self, name: str = "x", size: Union[int, Tuple[int, int]] = 1
    ) -> SX:
        """
        Create a new state with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new state will be vectorized (casadi.vec) to be
        included in the state vector (model.x).

        :param name: str
        :param size: int|tuple
        :return:
        """
        if callable(getattr(self, "name_variable", None)):
            name = self.name_variable(name)

        new_x = SX.sym(name, size)
        new_x_0 = SX.sym(name + "_0_sym", size)
        self.include_state(vec(new_x), ode=None, x_0=vec(new_x_0))
        return new_x

    def include_state(
        self, var: SX, ode: Optional[SX] = None, x_0: Optional[SX] = None
    ) -> SX:
        self.x = vertcat(self.x, var)

        if x_0 is None:
            x_0 = vertcat(*[SX.sym(var_i.name()) for var_i in var.nz])
        self.x_0 = vertcat(self.x_0, x_0)

        # crate entry for included state
        for x_i in var.nz:
            if x_i in self._ode:
                raise ValueError(f'State "{x_i}" already in this model')
            self._ode[x_i] = None
        if ode is not None:
            self.include_equations(ode=ode, x=var)
        return x_0

    def remove_state(self, var: SX):
        self.x = remove_variables_from_vector(var, self.x)

        for x_i in var.nz:
            del self._ode[x_i]

    def replace_variable(self, original: SX, replacement: SX):
        if original.numel() != replacement.numel():
            raise ValueError(
                "Original and replacement must have the same number of elements!"
                "original.numel()={}, replacement.numel()={}".format(
                    original.numel(), replacement.numel()
                )
            )

        if callable(getattr(super(), "replace_variable", None)):
            super().replace_variable(original, replacement)

        if original.numel() > 0:
            for x_i, x_i_eq in self._ode.items():
                if x_i_eq is not None:
                    self._ode[x_i] = substitute(x_i_eq, original, replacement)

    def include_equations(self, *args: SX, **kwargs: Union[SX, List[SX]]):
        """
        Differential equations can passed via `ode` kw argument with a `x` positional
        argument or via equallity in positional arguments.
        """
        if callable(getattr(super(), "include_equations", None)):
            super().include_equations(*args, **kwargs)

        if "ode" not in kwargs and "x" in kwargs:
            raise ValueError(
                "`x` is not given, but is `ode` given, what do I do with this `x`?"
            )

        if len(args) > 0:
            self._include_equation_from_diff_equations(*args)

        if "ode" not in kwargs:
            return
        ode = kwargs["ode"]
        x = kwargs.pop("x", None)

        # if is in the list form
        if isinstance(ode, abc.Sequence):
            ode = vertcat(*ode)

        if isinstance(x, abc.Sequence):
            x = vertcat(*x)

        # if ode was passed but not x, try to guess the x
        if x is None:
            # Check if None are all sequential, ortherwise we don't know who it belongs
            first_none = list(self._ode.values()).index(None)
            if any(eq is not None for eq in islice(self._ode.values(), 0, first_none)):
                raise ValueError(
                    "ODE should be inserted on the equation form or in the list form."
                    "You can't mix both without explicit passing the states associated with the equation."
                )
            x = vertcat(*list(self._ode.keys())[first_none : first_none + ode.numel()])

        self._include_equation_from_kwargs(x, ode)

    def _include_equation_from_kwargs(self, x: SX, ode: SX):
        #
        assert (
            ode.numel() == x.numel()
        ), f"Expected `x` and `ode` of same size, {x.numel()}!={ode.numel()}"

        for x_i in x.nz:
            if self._ode[x_i] is not None:
                warnings.warn(
                    f'State "{x_i}" already had an ODE associated, overriding it!'
                )
        self._ode = {**self._ode, **dict(zip(x.nz, ode.nz))}

    def _include_equation_from_diff_equations(self, *args: Union[SX, EqualityEquation]):
        x = SX(0, 1)
        ode = SX(0, 1)

        # get ode and x from equality equations
        for eq in args:
            if isinstance(eq, EqualityEquation) and isinstance(eq.lhs, Derivative):
                ode = vertcat(ode, eq.rhs)
                x = vertcat(x, eq.lhs.inner)

        # actually include the equations
        self._include_equation_from_kwargs(x, ode)
