from typing import List, Union

from casadi import DM, SX, Function, rootfinder, substitute, vec, vertcat

from yaocptool import config
from yaocptool.modelling.mixins.base_mixin import BaseMixin
from yaocptool.util.util import is_equality, remove_variables_from_vector


class AlgebraicMixin(BaseMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alg = SX()
        self.y = SX()

    @property
    def n_y(self) -> int:
        return self.y.numel()

    @property
    def y_names(self) -> List[str]:
        return [self.y[i].name() for i in range(self.n_y)]

    def create_algebraic_variable(self, name: str = "y", size: int = 1) -> SX:
        """
        Create a new algebraic variable with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new algebraic variable will be vectorized (casadi.vec)
        to be included in the algebraic vector (model.y).

        :param str name:
        :param int||tuple size:
        :return:
        """
        if callable(getattr(self, "name_variable", None)):
            name = self.name_variable(name)

        new_y = SX.sym(name, size)
        vec_y = vec(new_y)
        self.include_algebraic(vec_y)
        return new_y

    def find_algebraic_variable(
        self,
        x: DM,
        u: DM,
        guess: Union[DM, List[float]] = None,
        t: float = 0.0,
        p: Union[DM, List[float]] = None,
        theta_value: Union[DM, List[float]] = None,
        rootfinder_options=None,
    ) -> DM:
        if guess is None:
            guess = DM([1.0] * self.n_y)
        if isinstance(guess, list):
            guess = DM(guess)
        if rootfinder_options is None:
            rootfinder_options = dict(
                nlpsol="ipopt", nlpsol_options=config.SOLVER_OPTIONS["nlpsol_options"]
            )
        if p is None:
            p = []
        if theta_value is None:
            theta_value = []

        # replace known variables
        alg = self.alg
        known_var = vertcat(self.t, self.x, self.u, self.p, self.theta)
        known_var_values = vertcat(t, x, u, p, theta_value)
        alg = substitute(alg, known_var, known_var_values)

        f_alg = Function("f_alg", [self.y], [alg])

        rf = rootfinder("rf_algebraic_variable", "nlpsol", f_alg, rootfinder_options)
        return rf(guess)

    def include_algebraic(self, var: SX, alg: SX = None):
        self.y = vertcat(self.y, var)
        if alg is not None:
            self.include_equations(alg=alg)

    def remove_algebraic(self, var: SX, eq: SX = None):
        self.y = remove_variables_from_vector(var, self.y)
        if eq is not None:
            self.alg = remove_variables_from_vector(eq, self.alg)

    def replace_variable(self, original: SX, replacement: SX):
        # TODO: Remove or reinstate
        #  if isinstance(original, list):
        #      original = vertcat(*original)
        #  if isinstance(replacement, list):
        #      replacement = vertcat(*replacement)

        if original.numel() != replacement.numel():
            raise ValueError(
                "Original and replacement must have the same number of elements!"
                "original.numel()={}, replacement.numel()={}".format(
                    original.numel(), replacement.numel()
                )
            )

        if callable(getattr(super(), "replace_variable", None)):
            super().replace_variable(original, replacement)

        self.alg = substitute(self.alg, original, replacement)

    def include_equations(self, *args: SX, **kwargs: Union[SX, List[SX]]):
        """
        Algebraic equations can passed via `alg` kw argument or via equallity in positiona arguments.
        """
        if callable(getattr(super(), "include_equations", None)):
            super().include_equations(*args, **kwargs)

        alg = kwargs.pop("alg", SX(0, 1))
        if isinstance(alg, list):
            alg = vertcat(*alg)

        # in case a list of equations `y == x + u` has been passed
        for eq in args:
            if is_equality(eq):
                alg = vertcat(alg, eq.dep(0) - eq.dep(1))

        if alg is not None:
            self.alg = vertcat(self.alg, alg)
