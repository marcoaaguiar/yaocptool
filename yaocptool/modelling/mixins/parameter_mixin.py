from typing import List, Tuple, Union
from yaocptool.modelling.mixins.base_mixin import BaseMixin
from casadi import SX, vec, vertcat
from yaocptool.util.util import remove_variables_from_vector


class ParameterMixin(BaseMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.p = SX([])
        self.theta = SX([])

    @property
    def n_p(self) -> int:
        return self.p.numel()

    @property
    def n_theta(self) -> int:
        return self.theta.numel()

    @property
    def p_names(self) -> List[str]:
        return [self.p[i].name() for i in range(self.n_p)]

    @property
    def theta_names(self) -> List[str]:
        return [self.theta[i].name() for i in range(self.n_theta)]

    def create_parameter(
        self, name: str = "p", size: Union[int, Tuple[int, int]] = 1
    ) -> SX:
        """
        Create a new parameter name "name" and size "size"

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, "name_variable", None)):
            name = self.name_variable(name)

        new_p = SX.sym(name, size)
        self.include_parameter(vec(new_p))
        return new_p

    def create_theta(
        self, name: str = "p", size: Union[int, Tuple[int, int]] = 1
    ) -> SX:
        """
        Create a new parameter name "name" and size "size"

        :param name: str
        :param size: int
        :return:
        """
        if callable(getattr(self, "name_variable", None)):
            name = self.name_variable(name)

        new_theta = SX.sym(name, size)
        self.include_theta(vec(new_theta))
        return new_theta

    def include_parameter(self, p: SX):
        self.p = vertcat(self.p, p)

    def include_theta(self, theta: SX):
        self.theta = vertcat(self.theta, theta)

    def remove_parameter(self, var: SX):
        self.p = remove_variables_from_vector(var, self.p)

    def remove_theta(self, var: SX):
        self.theta = remove_variables_from_vector(var, self.theta)
