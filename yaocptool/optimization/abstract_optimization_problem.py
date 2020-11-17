from numbers import Number
from typing import Dict, Union

from casadi import vertcat, MX, inf, DM, repmat, is_equal, depends_on

from yaocptool import is_inequality, is_equality


class AbstractOptimizationProblem(object):
    def __init__(self, name: str = "optimization_problem", **kwargs):
        r"""
            Abstract Optimization Problem class
            Optimization problem

        .. math::
            \\min_x f(x, p)

            \\textrm{s.t.:} g_{lb} \leq g(x,p) \leq g_{ub}

            Object attributes:
            x -> optimization variables
            g -> constraint

        :param name: Optimization problem name
        """
        self.name = name

        self.f = MX([])
        self.g = MX([])
        self.x = MX([])
        self.p = DM([])

        self.g_lb = DM([])
        self.g_ub = DM([])
        self.x_lb = DM([])
        self.x_ub = DM([])

        self.solver_options = {}

        self._solver = None

        for (key, val) in kwargs.items():
            setattr(self, key, val)

    def create_variable(
        self,
        name: str,
        size: int = 1,
        lb: Union[DM, float] = -inf,
        ub: Union[DM, float] = inf,
    ) -> MX:
        """Create an optimization variable

        :param str name: Name of the optimization variable.
        :param int size: Size of the variable (default = 1)
        :param MX|SX lb: Lower bound of the variable. If the given 'size' is greater than one but a scalar is passed as
            lower bound, a vector of lb of size 'size' will be used as a lower bound. (default = [-inf]*size)
        :param MX|SX ub: Upper bound of the variable. If the given 'size' is greater than one but a scalar is passed as
            upper bound, a vector of ub of size 'size' will be used as a upper bound. (default = [inf]*size)
        :return: Return the variable
        :rtype: MX
        """
        if isinstance(lb, (float, int)):
            lb = repmat(lb, size)
        if isinstance(ub, (float, int)):
            ub = repmat(ub, size)

        new_x = MX.sym(name, size)

        self.include_variable(new_x, lb=lb, ub=ub)
        return new_x

    def include_variable(
        self, variable: MX, lb: Union[float, DM] = -inf, ub: Union[float, DM] = inf
    ):
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

        if variable.numel() != lb.numel() or variable.numel() != ub.numel():
            raise ValueError(
                "Lower bound or upper bound has different size of the given variable"
            )

        if (
            not lb.is_constant()
            and depends_on(lb, self.p)
            or not ub.is_constant()
            and depends_on(ub, self.p)
        ):
            raise ValueError(
                "Neither the lower or the upper bound can depend on the optimization problem parameter. "
                "lb={}, ub={}".format(lb, ub)
            )

        for i, (lb_i, var_i, ub_i) in enumerate(zip(lb.nz, variable.nz, ub.nz)):
            if lb_i > ub_i:
                raise ValueError(
                    "Lower bound is greater than upper bound for index {}. "
                    "The inequality {} <= {} <= is infeasible".format(
                        i, lb_i, var_i, ub_i
                    )
                )

        self.x = vertcat(self.x, variable)
        self.x_lb = vertcat(self.x_lb, lb)
        self.x_ub = vertcat(self.x_ub, ub)

    def create_parameter(self, name, size=1):
        """
            Create parameter for in the Optimization Problem

        :param str name: variable name
        :param int size: number of rows
        :return: created variable
        :rtype: MX|SX
        """
        new_p = MX.sym(name, size)
        self.include_parameter(new_p)
        return new_p

    def include_parameter(self, par):
        """
            Include parameter for in the Optimization Problem

        :param MX|SX par: parameter to be included
        """
        self.p = vertcat(self.p, par)

    def set_objective(self, expr):
        """
            Set objective function

        :param MX|SX expr: objective function
        """
        if isinstance(expr, list):
            expr = vertcat(*expr)
        if isinstance(expr, (float, int)):
            expr = vertcat(expr)

        if expr.numel() > 1:
            raise ValueError(
                "Objective function should be an scalar. "
                "Given objective has shape = {}".format(expr.shape)
            )
        self.f = expr

    def include_inequality(self, expr, lb=None, ub=None):
        """Include inequality to the problem with the following form
        lb <= expr <= ub

        :param expr: expression for the inequality, this is the only term that should contain symbolic variables
        :param lb: Lower bound of the inequality. If the 'expr' size is greater than one but a scalar is passed as
            lower bound, a vector of lb with size of 'expr' will be used as a lower bound. (default = [-inf]*size)
        :param ub: Upper bound of the inequality. If the  'expr' size is greater than one but a scalar is passed as
            upper bound, a vector of ub with size of  'expr' will be used as a upper bound. (default = [inf]*size)
        """
        # check expr
        if isinstance(expr, list):
            expr = vertcat(expr)
        if expr.size2() > 1:
            raise Exception(
                "Given expression is not a vector, number of columns is {}".format(
                    expr.size2()
                )
            )

        # check lower bound
        if lb is None:
            lb = -DM.inf(expr.size1())
        else:
            lb = vertcat(lb)
            if lb.numel() == 1 and expr.numel() > 1:
                lb = repmat(lb, expr.numel())

        # check lb correct size
        if expr.shape != lb.shape:
            raise ValueError(
                "Expression and lower bound does not have the same size: "
                "expr.shape={}, lb.shape=={}".format(expr.shape, lb.shape)
            )
        # check upper bound
        if ub is None:
            ub = DM.inf(expr.size1())
        else:
            ub = vertcat(ub)
            if ub.numel() == 1 and expr.numel() > 1:
                ub = repmat(ub, expr.numel())

        # check ub correct size
        if expr.shape != ub.shape:
            raise ValueError(
                "Expression and lower bound does not have the same size: "
                "expr.shape={}, lb.shape=={}".format(expr.shape, ub.shape)
            )

        # check for if lb or ub have 'x's and 'p's
        if depends_on(vertcat(lb, ub), vertcat(self.x, self.p)):
            raise ValueError(
                "The lower and upper bound cannot contain variables from the optimization problem."
                "LB: {}, UB: {}".format(lb, ub)
            )

        for i in range(expr.numel()):
            if lb.is_constant() and ub.is_constant() and lb[i] > ub[i]:
                raise ValueError(
                    "Lower bound is greater than upper bound for index {}. "
                    "The inequality {} <= {} <= is infeasible".format(
                        i, lb[i], expr[i], ub[i]
                    )
                )

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
            raise Exception(
                "Given expression is not a vector, number of columns is {}".format(
                    expr.size2()
                )
            )

        if rhs is None:
            rhs = DM.zeros(expr.shape)
        else:
            rhs = vertcat(rhs)
            if rhs.numel() == 1 and expr.numel() > 1:
                rhs = repmat(rhs, expr.numel())

            if expr.shape != rhs.shape:
                msg = (
                    "Expression and the right hand side does not have the same size: "
                    "expr.shape={}, rhs.shape=={}".format(expr.shape, rhs.shape)
                )
                raise ValueError(msg)

        # check for if rhs have 'x's and 'p's
        if depends_on(rhs, vertcat(self.x, self.p)):
            raise ValueError(
                "Right-hand side cannot contain variables from the optimization problem. "
                "RHS = {}".format(rhs)
            )

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, rhs)
        self.g_ub = vertcat(self.g_ub, rhs)

    def include_constraint(self, expr):
        """Includes an inequality or inequality to the optimization problem,
        Example:
        opt_problem.include_constraint(1 <= x**2)
        opt_problem.include_constraint(x + y == 1)
        Due to limitations on CasADi it does not allows for double inequalities (e.g.: 0 <= x <= 1)

        :param casadi.MX expr: equality or inequality expression
        """
        # Check for inconsistencies
        if not is_inequality(expr) and not is_equality(expr):
            raise ValueError(
                "The passed 'expr' was not recognized as an equality or inequality constraint"
            )
        if expr.dep(0).is_constant() and expr.dep(1).is_constant():
            raise ValueError("Both sides of the constraint are constant")
        if depends_on(expr.dep(0), vertcat(self.x, self.p)) and depends_on(
            expr.dep(1), vertcat(self.x, self.p)
        ):
            raise ValueError(
                "One of the sides of the constraint cannot depend on the problem. (e.g.: x<=1)"
                "lhs: {}, rhs: {}".format(expr.dep(0), expr.dep(1))
            )

        # find the dependent and independent term
        if depends_on(expr.dep(0), vertcat(self.x, self.p)):
            dep_term = expr.dep(0)
            indep_term = expr.dep(1)
        else:
            dep_term = expr.dep(1)
            indep_term = expr.dep(0)
        if indep_term.is_constant():
            indep_term = indep_term.to_DM()
        # if it is and equality
        if is_equality(expr):
            self.include_equality(dep_term, indep_term)

        # if it is an inequality
        elif is_inequality(expr):
            # by default all inequalities are treated as 'less than' or 'less or equal', e.g.: x<=1 or 1<=x
            # if the term on the rhs term is the non-symbolic term, then it is a 'x<=1' inequality,
            # where the independent term is the upper bound
            if is_equal(indep_term, expr.dep(1)):
                self.include_inequality(dep_term, ub=indep_term)
            # otherwise, it is a '1<=x' inequality, where the independent term is the lower bound
            else:
                self.include_inequality(dep_term, lb=indep_term)

    def get_problem_dict(self) -> Dict[str, Union[MX, DM]]:
        """Return the optimization problem in a Python dict form (CasADi standard).
        The dictionary keys are:
        f-> objective function
        g-> constraints
        x-> variables
        p-> parameters

        :return: optimization problem as a dict
        :rtype: dict
        """
        return {"f": self.f, "g": self.g, "x": self.x, "p": self.p}

    def get_solver(self):
        """
            Get optimization solver

        :return:
        """
        if self._solver is None:
            self._solver = self._create_solver()
        return self._solver

    def _create_solver(self):
        """
            create optimization solver

        :rtype: casadi.nlpsol
        """
        raise NotImplementedError

    def get_default_call_dict(self):
        """
        Return a dictionary of the settings that will be used on calling the solver
        The keys are:
        - 'lbx': lower bound on the variables
        - 'ubx': upper bound on the variables
        - 'lbg': lower bound on the constraints
        - 'ubg': upper bound on the constraints

        :return: dict with default values
        :rtype: dict
        """
        return {"lbx": self.x_lb, "ubx": self.x_ub, "lbg": self.g_lb, "ubg": self.g_ub}

    def solve(self, initial_guess, call_dict=None, p=None, lam_x=None, lam_g=None):
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
            call_dict["x0"] = initial_guess
        if lam_x is not None:
            call_dict["lam_x0"] = lam_x
        if lam_g is not None:
            call_dict["lam_g0"] = lam_g

        call_dict["p"] = vertcat(p)

        if call_dict["p"].numel() != self.p.numel():
            raise ValueError(
                'Passed parameter "p" has size {}, '
                "while problem.p has size {}".format(
                    call_dict["p"].numel(), self.p.numel()
                )
            )

        solver = self.get_solver()

        solution = solver(**call_dict)
        solution["stats"] = solver.stats()

        return solution
