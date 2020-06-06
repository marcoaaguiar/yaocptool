from casadi.casadi import is_equal, substitute, SX, vec, vertcat
from yaocptool.util.util import find_variables_indices_in_vector, remove_variables_from_vector, remove_variables_from_vector_by_indices


class ControlMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.u = SX([])

        self._parametrized_controls = []
        self.u_par = vertcat(self.u)
        self.u_expr = vertcat(self.u)

    @property
    def n_u(self):
        return self.u.numel()

    def create_control(self, name="u", size=1):
        """
        Create a new control variable name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new control variable will be vectorized (casadi.vec)
        to be included in the control vector (model.u).

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, 'name_variable', None)):
            name = self.name_variable(name)

        new_u = SX.sym(name, size)
        self.include_control(vec(new_u))
        return new_u

    def include_control(self, var):
        self.u = vertcat(self.u, var)
        self.u_expr = vertcat(self.u_expr, var)
        self.u_par = vertcat(self.u_par, var)

    def remove_control(self, var):
        self.u_expr = remove_variables_from_vector_by_indices(
            find_variables_indices_in_vector(var, self.u), self.u_expr)
        self.u = remove_variables_from_vector(var, self.u)
        self.u_par = remove_variables_from_vector(var, self.u_par)

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

        #  self.u_par = substitute(self.u_par, original, replacement)
        self.u_expr = substitute(self.u_expr, original, replacement)

    def parametrize_control(self, u, expr, u_par=None):
        """
            Parametrize a control variables so it is a function of a set of parameters or other model variables.

        :param list|casadi.SX u:
        :param list|casadi.SX expr:
        :param list|casadi.SX u_par:
        """
        # input check
        if isinstance(u, list):
            u = vertcat(*u)
        if isinstance(u_par, list):
            u_par = vertcat(*u_par)
        if isinstance(expr, list):
            expr = vertcat(*expr)

        if not u.numel() == expr.numel():
            raise ValueError(
                "Passed control and parametrization expression does not have same size. "
                "u ({}) and expr ({})".format(u.numel(), expr.numel()))

        # Check and register the control parametrization.
        for u_i in u.nz:
            if self.control_is_parametrized(u_i):
                raise ValueError(
                    f'The control "{u_i}" is already parametrized.')
            # to get have a new memory address
            self._parametrized_controls = self._parametrized_controls + [u_i]

        # Remove u from u_par if they are going to be parametrized
        self.u_par = remove_variables_from_vector(u, self.u_par)
        if u_par is not None:
            self.u_par = vertcat(self.u_par, u_par)

        # Replace u by expr into the system
        self.replace_variable(u, expr)

    def create_input(self, name="u", size=1):
        """
        Same as the "model.create_control" function.
        Create a new control/input variable name "name" and size "size".
        Size can be an int or a tuple (e.g. (2,2)). However, the new control variable will be vectorized (casadi.vec)
        to be included in the control vector (model.u).

        :param name: str
        :param size: int
        :return:
        """
        return self.create_control(name, size)

    def control_is_parametrized(self, u):
        """
            Check if  the control "u" is parametrized

        :param casadi.SX u:
        :rtype bool:
        """
        u = vertcat(u)
        if not u.numel() == 1:
            raise ValueError(
                'The parameter "u" is expected to be of size 1x1, given: {}x{}'
                .format(*u.shape))
        if any([
                is_equal(u, parametrized_u)
                for parametrized_u in self._parametrized_controls
        ]):
            return True
        return False
