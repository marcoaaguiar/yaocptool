from numbers import Number

from casadi import vertcat, MX, inf, DM


class AbstractOptimizationProblem(object):
    def __init__(self, **kwargs):
        """ Abstract Optimization Problem class
            Optimization problem

            minimize f(x)
               x
            subject to: g_eq = 0
                        g
            Object attributes:
            x -> optimization variables
            g -> constraint

        :param n_x: int
        """
        self.name = 'optimization_problem'

        self.f = []
        self.g = []
        self.x = []
        self.p = []

        self.g_lb = []
        self.g_ub = []
        self.x_lb = []
        self.x_ub = []

        self.solver_options = {}

        self._solver = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def create_variable(self, name, size=1, lb=-inf, ub=inf):
        """Create an optimization variable

        :param str name: Name of the optimization variable.
        :param int size: Size of the variable (default = 1)
        :param lb: Lower bound of the variable. If the given 'size' is greater than one but a scalar is passed as lower
        bound, a vector of lb of size 'size' will be used as a lower bound. (default = [-inf]*size)
        :param ub: Upper bound of the variable. If the given 'size' is greater than one but a scalar is passed as upper
        bound, a vector of ub of size 'size' will be used as a upper bound. (default = [inf]*size)
        :return: Return the variable
        :rtype: MX
        """
        if isinstance(lb, Number):
            lb = [lb] * size
        if isinstance(ub, Number):
            ub = [ub] * size

        new_x = MX.sym(name, size)
        self.x = vertcat(self.x, new_x)
        self.x_lb = vertcat(self.x_lb, lb)
        self.x_ub = vertcat(self.x_ub, ub)
        return new_x

    def create_parameter(self, name, size=1):
        new_p = MX.sym(name, size)
        self.p = vertcat(self.p, new_p)
        return new_p

    def set_objective(self, expr):
        self.f = expr

    def include_inequality(self, expr, lb=None, ub=None):
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception("Given expression is not a vector, number of columns is {}".format(expr.size2()))
        if lb is None:
            lb = -inf * DM.ones(expr.size1())
        else:
            if not expr.shape == lb.shape:
                msg = "Expression and lower bound does not have the same size: expr.shape={}, lb.shape=={}".format(
                    expr.shape, lb.shape)
                raise ValueError(msg)
        if ub is None:
            ub = inf * DM.ones(expr.size1())
        else:
            if not expr.shape == ub.shape:
                msg = "Expression and upper bound does not have the same size: expr.shape={}, ub.shape=={}".format(
                    expr.shape, ub.shape)
                raise ValueError(msg)

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, lb)
        self.g_ub = vertcat(self.g_ub, ub)

    def include_equality(self, expr, rhs=None):
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception("Given expression is not a vector, number of columns is {}".format(expr.size2()))

        if rhs is None:
            rhs = DM.zeros(expr.shape)
        else:
            if not expr.shape == rhs.shape:
                msg = "Expression and the right hand side does not have the same size: " \
                      "expr.shape={}, rhs.shape=={}".format(expr.shape, rhs.shape)
                raise ValueError(msg)

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, rhs)
        self.g_ub = vertcat(self.g_ub, rhs)

    def get_problem_dict(self):
        return {'f': self.f, 'g': self.g, 'x': self.x, 'p': self.p}

    def get_solver(self):
        if self._solver is None:
            self._solver = self._create_solver()
        return self._solver

    def _create_solver(self):
        raise NotImplementedError

    def get_default_call_dict(self):
        return {'lbx': self.x_lb,
                'ubx': self.x_ub,
                'lbg': self.g_lb,
                'ubg': self.g_ub}

    def solve(self, call_dict=None, p=None):
        if call_dict is None:
            call_dict = self.get_default_call_dict()
        if p is None:
            p = []
        call_dict = dict(call_dict)
        call_dict['p'] = p
        solver = self.get_solver()

        return solver(**call_dict)
