from casadi.casadi import SX, vec, vertcat
from yaocptool.util.util import remove_variables_from_vector


class ParameterMixin:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = SX([])
        self.theta = SX([])

    @property
    def n_p(self):
        return self.p.numel()

    @property
    def n_theta(self):
        return self.theta.numel()

    @property
    def p_names(self):
        return [self.p[i].name() for i in range(self.n_p)]

    @property
    def theta_names(self):
        return [self.theta[i].name() for i in range(self.n_theta)]

    def create_parameter(self, name="p", size=1):
        """
        Create a new parameter name "name" and size "size"

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, 'name_variable', None)):
            name = self.name_variable(name)

        new_p = SX.sym(name, size)
        self.include_parameter(vec(new_p))
        return new_p

    def create_theta(self, name="theta", size=1):
        """
        Create a new parameter name "name" and size "size"

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, 'name_variable', None)):
            name = self.name_variable(name)

        new_theta = SX.sym(name, size)
        self.include_theta(vec(new_theta))
        return new_theta

    def include_parameter(self, p):
        self.p = vertcat(self.p, p)

    def include_theta(self, theta):
        self.theta = vertcat(self.theta, theta)

    def remove_parameter(self, var):
        self.p = remove_variables_from_vector(var, self.p)

    def remove_theta(self, var):
        self.theta = remove_variables_from_vector(var, self.theta)
