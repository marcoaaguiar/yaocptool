from numbers import Number

from casadi import vertcat, MX, inf, DM, repmat


class AbstractOptimizationProblem(object):
    def __init__(self, **kwargs):
        """
            Abstract Optimization Problem class
            Optimization problem

        .. math::
            \\min_x f(x, p)

            \\textrm{s.t.:} g_{lb} \leq g(x,p) \leq g_{ub}

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
            lb = repmat(lb, size)
        if isinstance(ub, Number):
            ub = repmat(ub, size)

        new_x = MX.sym(name, size)

        self.include_variable(new_x, lb=lb, ub=ub)
        return new_x

    def include_variable(self, variable, lb=-inf, ub=inf):
        """Include a symbolic variable in the optimization problem

        :param variable: variable to be included
        :param lb: Lower bound of the variable. If the given variable size is greater than one but a scalar is passed as
            lower bound, a vector of lb with size of the given variable will be used as a lower bound.
            (default = [-inf]*size)
        :param ub: Upper bound of the variable. If the given variable size is greater than one but a scalar is passed as
            upper bound, a vector of ub  with size of the given variable will be used as a upper bound.
            (default = [inf]*size)
        """
        lb = vertcat(lb)
        ub = vertcat(ub)

        if lb.numel() == 1 and variable.numel() > 1:
            lb = repmat(lb, variable.numel())
        if ub.numel() == 1 and variable.numel() > 1:
            ub = repmat(ub, variable.numel())

        if not variable.numel() == lb.numel() or not variable.numel() == ub.numel():
            raise Exception("Lower bound or upper bound has different size of the given variable")

        self.x = vertcat(self.x, variable)
        self.x_lb = vertcat(self.x_lb, lb)
        self.x_ub = vertcat(self.x_ub, ub)

    def create_parameter(self, name, size=1):
        new_p = MX.sym(name, size)
        self.include_parameter(new_p)
        return new_p

    def include_parameter(self, par):
        self.p = vertcat(self.p, par)

    def set_objective(self, expr):
        self.f = expr

    def include_inequality(self, expr, lb=None, ub=None):
        """ Include inequality to the problem with the following form
        lb <= expr <= ub

        :param expr: expression for the inequality, this is the only term that should contain symbolic variables
        :param lb: Lower bound of the inequality. If the 'expr' size is greater than one but a scalar is passed as
            lower bound, a vector of lb with size of 'expr' will be used as a lower bound. (default = [-inf]*size)
        :param ub: Upper bound of the inequality. If the  'expr' size is greater than one but a scalar is passed as
            upper bound, a vector of ub with size of  'expr' will be used as a upper bound. (default = [inf]*size)
        """
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception("Given expression is not a vector, number of columns is {}".format(expr.size2()))
        if lb is None:
            lb = -inf * DM.ones(expr.size1())
        else:
            lb = vertcat(lb)
            if lb.numel() == 1 and expr.numel() > 1:
                lb = repmat(lb, expr.numel())

            if not expr.shape == lb.shape:
                msg = "Expression and lower bound does not have the same size: expr.shape={}, lb.shape=={}".format(
                    expr.shape, lb.shape)
                raise ValueError(msg)
        if ub is None:
            ub = inf * DM.ones(expr.size1())
        else:
            ub = vertcat(ub)
            if ub.numel() == 1 and expr.numel() > 1:
                ub = repmat(ub, expr.numel())
            if not expr.shape == ub.shape:
                msg = "Expression and upper bound does not have the same size: expr.shape={}, ub.shape=={}".format(
                    expr.shape, ub.shape)
                raise ValueError(msg)

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, lb)
        self.g_ub = vertcat(self.g_ub, ub)

    def include_equality(self, expr, rhs=None):
        """Include a equality with the following form
        expr = rhs

        :param expr: expression, this is the only term that should contain symbolic variables
        :param rhs: right hand side, by default it is a vector of zeros with same size of expr. If the  'expr' size is
            greater than one but a scalar is passed as 'rhs', a vector of 'rhs' with size of 'expr' will be used as
            right hand side. (default = [0]*size)
        """
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception("Given expression is not a vector, number of columns is {}".format(expr.size2()))

        if rhs is None:
            rhs = DM.zeros(expr.shape)
        else:
            rhs = vertcat(rhs)
            if rhs.numel() == 1 and expr.numel() > 1:
                rhs = repmat(rhs, expr.numel())

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

    def solve(self, initial_guess=None, call_dict=None, p=None, lam_x=None, lam_g=None):
        """

        :param initial_guess: Initial guess
        :param call_dict: a dictionary containing 'lbx', 'ubx', 'lbg', 'ubg'. If not given, the one obtained with
            self.get_default_call_dict will be used.
        :param p: parameters
        :param lam_x:
        :param lam_g:
        :return: dictionary with solution
        """
        if call_dict is None:
            call_dict = self.get_default_call_dict()
        if p is None:
            p = []
        call_dict = dict(call_dict)
        if initial_guess is not None:
            call_dict['x0'] = initial_guess
        if lam_x is not None:
            call_dict['lam_x0'] = lam_x
        if lam_g is not None:
            call_dict['lam_g0'] = lam_g

        call_dict['p'] = p
        solver = self.get_solver()

        return solver(**call_dict)
