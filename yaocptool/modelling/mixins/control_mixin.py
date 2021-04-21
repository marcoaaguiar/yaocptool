from typing import List, Optional, Tuple, Union

from casadi import SX, is_equal, substitute, vec, vertcat

from yaocptool.modelling.mixins.base_mixin import BaseMixin
from yaocptool.util import (
    find_variables_indices_in_vector,
    remove_variables_from_vector,
    remove_variables_from_vector_by_indices,
)


class ControlMixin(BaseMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u = SX(0, 1)

        self._parametrized_controls: List[SX] = []
        self.u_par = vertcat(self.u)
        self.u_expr = vertcat(self.u)

    @property
    def n_u(self) -> int:
        return self.u.numel()

    @property
    def n_u_par(self) -> int:
        return self.u_par.numel()

    @property
    def u_names(self) -> List[str]:
        return [self.u[i].name() for i in range(self.n_u)]

    def create_control(
        self, name: str = "u", size: Union[int, Tuple[int, int]] = 1
    ) -> SX:
        """
        Create a new control variable name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new control variable will be vectorized (casadi.vec)
        to be included in the control vector (model.u).

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, "name_variable", None)):
            name = self.name_variable(name)

        new_u = SX.sym(name, size)
        self.include_control(vec(new_u))
        return new_u

    def include_control(self, var: SX):
        self.u = vertcat(self.u, var)
        self.u_expr = vertcat(self.u_expr, var)
        self.u_par = vertcat(self.u_par, var)

    def remove_control(self, var: SX):
        self.u_expr = remove_variables_from_vector_by_indices(
            find_variables_indices_in_vector(var, self.u), self.u_expr
        )
        self.u = remove_variables_from_vector(var, self.u)
        self.u_par = remove_variables_from_vector(var, self.u_par)

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

        #  self.u_par = substitute(self.u_par, original, replacement)
        self.u_expr = substitute(self.u_expr, original, replacement)

    def parametrize_control(self, u: SX, expr: SX, u_par: Optional[SX] = None):
        """
            Parametrize a control variables so it is a function of a set of parameters or other model variables.

        :param list|casadi.SX u:
        :param list|casadi.SX expr:
        :param list|casadi.SX u_par:
        """
        if u.numel() != expr.numel():
            raise ValueError(
                "Passed control and parametrization expression does not have same size. "
                "u ({}) and expr ({})".format(u.numel(), expr.numel())
            )

        # Check and register the control parametrization.
        for u_i in u.nz:
            if self.control_is_parametrized(u_i):
                raise ValueError(f'The control "{u_i}" is already parametrized.')
            # to get have a new memory address
            self._parametrized_controls = self._parametrized_controls + [u_i]

        # Remove u from u_par if they are going to be parametrized
        self.u_par = remove_variables_from_vector(u, self.u_par)
        if u_par is not None:
            self.u_par = vertcat(self.u_par, u_par)

        # Replace u by expr into the system
        self.replace_variable(u, expr)

    def create_input(
        self, name: str = "u", size: Union[int, Tuple[int, int]] = 1
    ) -> SX:
        """
        Same as the "model.create_control" function.
        Create a new control/input variable name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new control variable will be vectorized (casadi.vec)
        to be included in the control vector (model.u).

        :param name:
        :param size:
        :return:
        """
        return self.create_control(name, size)

    def control_is_parametrized(self, u: SX) -> bool:
        """
            Check if  the control "u" is parametrized

        :param casadi.SX u:
        :rtype bool:
        """
        if u.numel() != 1:
            raise ValueError(
                'The parameter "u" is expected to be of size 1x1, given: {}x{}'.format(
                    *u.shape
                )
            )
        return any(
            is_equal(u, parametrized_u)
            for parametrized_u in self._parametrized_controls
        )
