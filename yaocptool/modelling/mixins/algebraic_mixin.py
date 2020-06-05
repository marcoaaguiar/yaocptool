from casadi.casadi import Function, rootfinder, substitute, SX, vec, vertcat
from distutils.command.config import config
from yaocptool.util.util import remove_variables_from_vector


class AlgebraicMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.alg = SX([])
        self.y = SX([])

    @property
    def n_y(self):
        return self.y.numel()

    def create_algebraic_variable(self, name="y", size=1):
        """
        Create a new algebraic variable with the name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new algebraic variable will be vectorized (casadi.vec)
        to be included in the algebraic vector (model.y).

        :param str name:
        :param int||tuple size:
        :return:
        """
        if self.model_name_as_prefix:
            name = self.name + "_" + name

        new_y = SX.sym(name, size)
        self.include_algebraic(vec(new_y))
        return new_y

    def find_algebraic_variable(self,
                                x,
                                u,
                                guess=None,
                                t=0.0,
                                p=None,
                                theta_value=None,
                                rootfinder_options=None):
        if guess is None:
            guess = [1] * self.n_y
        if rootfinder_options is None:
            rootfinder_options = dict(
                nlpsol="ipopt",
                nlpsol_options=config.SOLVER_OPTIONS["nlpsol_options"])
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

        rf = rootfinder("rf_algebraic_variable", "nlpsol", f_alg,
                        rootfinder_options)
        res = rf(guess)
        return res

    def include_algebraic(self, var, alg=None):
        self.y = vertcat(self.y, var)
        self.include_equations(alg=alg)

    def remove_algebraic(self, var, eq=None):
        self.y = remove_variables_from_vector(var, self.y)
        if eq is not None:
            self.alg = remove_variables_from_vector(eq, self.alg)
