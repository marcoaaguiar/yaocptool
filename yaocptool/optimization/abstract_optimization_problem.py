from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, TypedDict, Union

import ray
from casadi import (
    DM,
    Function,
    MX,
    StringSerializer,
    StringDeserializer,
    depends_on,
    inf,
    is_equal,
    repmat,
    vertcat,
)

from yaocptool import is_equality, is_inequality


class OptiResultDictType(TypedDict):
    f: DM
    g: DM
    lam_g: DM
    lam_p: DM
    lam_x: DM
    x: DM


class NlpSolProcess(Process):
    def __init__(self, pipe: Connection, solver: Function):
        super().__init__()
        self.pipe = pipe
        self.solver = solver

    def run(self) -> None:
        while True:
            kwarg = self.pipe.recv()
            if kwarg == "TERM":
                return
            result = self.solver(**kwarg)
            self.pipe.send(result)


class ExtendedOptiResultDictType(OptiResultDictType):
    stats: Dict[str, Any]


class AbstractOptimizationProblem(object):
    def __init__(self, name: str = "optimization_problem", **kwargs: Any):
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

        self.f: MX = MX()
        self.g: MX = MX()
        self.x: MX = MX()
        self.p: MX = MX()

        self.g_lb = DM()
        self.g_ub = DM()
        self.x_lb = DM()
        self.x_ub = DM()

        self.solver_options: Dict[str, Any] = {}

        self._solver: Optional[Function] = None
        self._connection: Optional[Connection] = None
        self._process: Optional[NlpSolProcess] = None

        for (key, val) in kwargs.items():
            setattr(self, key, val)

    def print_var(self):
        return vars(self)

    #  def __getstate__(self):
    #      variable_serializer = StringSerializer()
    #      bound_serializer = StringSerializer()
    #      #  solver_serializer = StringSerializer()
    #
    #      variable_serializer.pack([self.f, self.g, self.x, self.p])
    #      bound_serializer.pack([self.g_lb, self.g_ub, self.x_lb, self.x_ub])
    #      #  solver_serializer.pack(self._solver)
    #
    #      return {
    #          "name": self.name,
    #          "solver_options": self.solver_options,
    #          "variable_string": variable_serializer.encode(),
    #          "bounds_string": bound_serializer.encode(),
    #          #  "solver_string": solver_serializer.encode(),
    #          "_solver": self._solver,
    #      }
    #
    #  def __setstate__(self, state):
    #      self.f, self.g, self.x, self.p = StringDeserializer(
    #          state.pop("variable_string")
    #      ).unpack()
    #
    #      self.g_lb, self.g_ub, self.x_lb, self.x_ub = StringDeserializer(
    #          state.pop("bounds_string")
    #      ).unpack()
    #
    #      #  self._solver = StringDeserializer(state.pop("solver_string")).unpack()
    #      self.solver_options = state["solver_options"]
    #      self.name = state["name"]
    #      self._connection = None
    #      self._process = None
    #      self._solver = state["_solver"]

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
        if isinstance(lb, float):
            lb = vertcat(lb)
        if isinstance(ub, float):
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
                    "The inequality {} <= {} <= {} is infeasible".format(
                        i, lb_i, var_i, ub_i
                    )
                )

        self.x = vertcat(self.x, variable)
        self.x_lb = vertcat(self.x_lb, lb)
        self.x_ub = vertcat(self.x_ub, ub)

    def create_parameter(self, name: str, size: int = 1) -> MX:
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

    def include_parameter(self, par: MX):
        """
            Include parameter for in the Optimization Problem

        :param MX|SX par: parameter to be included
        """
        self.p = vertcat(self.p, par)

    def set_objective(self, expr: Union[MX, List[MX], float, int]):
        """
            Set objective function

        :param MX|SX expr: objective function
        """
        if isinstance(expr, list):
            expr = vertcat(*expr)
        if isinstance(expr, (int, float)):
            expr = MX(expr)

        if expr.numel() > 1:
            raise ValueError(
                "Objective function should be an scalar. "
                "Given objective has shape = {}".format(expr.shape)
            )
        self.f = expr

    def include_inequality(
        self,
        expr: MX,
        lb: Optional[Union[DM, float, List[float]]] = None,
        ub: Optional[Union[DM, float, List[float]]] = None,
    ):
        """Include inequality to the problem with the following form
        lb <= expr <= ub

        :param expr: expression for the inequality, this is the only term that should contain symbolic variables
        :param lb: Lower bound of the inequality. If the 'expr' size is greater than one but a scalar is passed as
            lower bound, a vector of lb with size of 'expr' will be used as a lower bound. (default = [-inf]*size)
        :param ub: Upper bound of the inequality. If the  'expr' size is greater than one but a scalar is passed as
            upper bound, a vector of ub with size of  'expr' will be used as a upper bound. (default = [inf]*size)
        """
        if isinstance(lb, (int, float)):
            lb = DM(lb)
        if isinstance(ub, (int, float)):
            ub = DM(ub)

        # check expr
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
                    "The inequality {} <= {} <= {} is infeasible".format(
                        i, lb[i], expr[i], ub[i]
                    )
                )

        self.g = vertcat(self.g, expr)
        self.g_lb = vertcat(self.g_lb, lb)
        self.g_ub = vertcat(self.g_ub, ub)

    def include_equality(self, expr: MX, rhs: Optional[Union[DM, float]] = None):
        """Include a equality with the following form
        expr = rhs

        :param expr: expression, this is the only term that should contain symbolic variables
        :param rhs: right hand side, by default it is a vector of zeros with same size of expr. If the  'expr' size is
            greater than one but a scalar is passed as 'rhs', a vector of 'rhs' with size of 'expr' will be used as
            right hand side. (default = [0]*size)
        """
        if expr.size2() > 1:
            raise Exception(
                "Given expression is not a vector, number of columns is {}".format(
                    expr.size2()
                )
            )

        if rhs is None:
            rhs = DM.zeros(expr.shape)
        else:
            if isinstance(rhs, (int, float)):
                rhs = DM(rhs)
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

    def include_constraint(self, expr: MX):
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
            indep_term_dm = indep_term.to_DM()
            # if it is and equality
            if is_equality(expr):
                self.include_equality(dep_term, indep_term_dm)

            # if it is an inequality
            elif is_inequality(expr):
                # by default all inequalities are treated as 'less than' or 'less or equal', e.g.: x<=1 or 1<=x
                # if the term on the rhs term is the non-symbolic term, then it is a 'x<=1' inequality,
                # where the independent term is the upper bound
                if is_equal(indep_term_dm, expr.dep(1)):
                    self.include_inequality(dep_term, ub=indep_term_dm)
                # otherwise, it is a '1<=x' inequality, where the independent term is the lower bound
                else:
                    self.include_inequality(dep_term, lb=indep_term_dm)
        else:
            raise NotImplementedError

    def get_problem_dict(self) -> Dict[str, Union[MX]]:
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

    def get_solver(self) -> Function:
        """
            Get optimization solver

        :return:
        """
        if self._solver is None:
            self._solver = self._create_solver()
        return self._solver

    def _create_solver(self) -> Function:
        """
            create optimization solver

        :rtype: casadi.nlpsol
        """
        raise NotImplementedError

    def get_default_call_dict(self) -> Dict[str, DM]:
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

    def solve(
        self,
        initial_guess: DM,
        call_dict: Optional[Dict[str, DM]] = None,
        p: Optional[DM] = None,
        lam_x: Optional[DM] = None,
        lam_g: Optional[DM] = None,
    ) -> ExtendedOptiResultDictType:
        """

        :param initial_guess: Initial guess
        :param call_dict: a dictionary containing 'lbx', 'ubx', 'lbg', 'ubg'. If not given, the one obtained with
            self.get_default_call_dict will be used.
        :param p: parameters
        :param lam_x:
        :param lam_g:
        :return: dictionary with solution
        """
        call_dict = self._get_solver_call_args(
            initial_guess, call_dict, p, lam_x, lam_g
        )

        solver = self.get_solver()

        solution = solver(**call_dict)
        solution["stats"] = solver.stats()
        print("solved")

        return solution

    def mp_solve(
        self,
        initial_guess: DM,
        call_dict: Optional[Dict[str, DM]] = None,
        p: Optional[DM] = None,
        lam_x: Optional[DM] = None,
        lam_g: Optional[DM] = None,
    ):
        if self._process is None:
            self._connection, conn_child = Pipe()
            self._process = NlpSolProcess(conn_child, self.get_solver())
            self._process.start()
        call_dict = self._get_solver_call_args(
            initial_guess, call_dict, p, lam_x, lam_g
        )
        self._connection.send(call_dict)

    def mp_get_solution(self):
        return self._connection.recv()

    def mp_terminate(self):
        self._connection.send("TERM")

    def _get_solver_call_args(self, initial_guess, call_dict, p, lam_x, lam_g):
        if call_dict is None:
            call_dict = self.get_default_call_dict()
        if p is None:
            p = DM()

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
        return call_dict
