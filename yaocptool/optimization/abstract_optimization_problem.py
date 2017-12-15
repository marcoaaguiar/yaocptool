from numbers import Number

from casadi import vertcat, MX, inf, DM


class AbstractOptimizationProblem(object):
    def __init__(self, n_x=0, **kwargs):
        """ Abstract Optimization Problem class
            Optimization problem

            minimize f(x)
               x
            subject to: g_eq = 0
                        g
            Object attributes:
            x -> optimization variables
            g -> contraint

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

    def create_variable(self, name, size, lb=-inf, ub=inf):
        if isinstance(lb, Number):
            lb = [lb] * size
        if isinstance(ub, Number):
            ub = [ub] * size

        new_x = MX.sym(name, size)
        self.x = vertcat(self.x, new_x)
        self.x_lb = vertcat(self.x_lb, lb)
        self.x_ub = vertcat(self.x_ub, ub)
        return new_x

    def create_parameter(self, name, size):
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

        if ub is None:
            ub = inf * DM.ones(expr.size1())

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, lb)
        self.g_ub = vertcat(self.g_ub, ub)

    def include_equality(self, expr, lhs=0):
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception("Given expression is not a vector, number of columns is {}".format(expr.size2()))

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, lhs)
        self.g_ub = vertcat(self.g_ub, lhs)

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
