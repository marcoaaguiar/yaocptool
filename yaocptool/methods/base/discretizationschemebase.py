from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, overload

from casadi import DM, MX, SX, Function, vertcat

from yaocptool import convert_expr_from_tau_to_time
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.modelling.ocp import OptimalControlProblem
from yaocptool.modelling.system_model import SystemModel
from yaocptool.optimization.abstract_optimization_problem import (
    AbstractOptimizationProblem,
)

if TYPE_CHECKING:
    from casadi import NumericMT

    from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
else:
    SolutionMethodsBase = "SolutionMethodsBase"
    NumericMT = "NumericMT"


class DiscretizationSchemeBase:
    def __init__(self, solution_method: SolutionMethodsBase):
        """Base class for discretization methods. A discretization class transforms and OCP into a NLP

        :type solution_method: yaocptool.methods.base.solutionmethodsbase.SolutionMethodsBase
        """
        self.solution_method = solution_method

        if (
            self.solution_method.degree_control > 1
            and self.solution_method.problem.has_delta_u
        ):
            raise Exception(
                'Maximum and minimum value for Delta u only defined for "degree_control" == 1. '
                'Current "degree_control":{}'.format(
                    self.solution_method.degree_control
                )
            )

    @property
    def model(self) -> SystemModel:
        """

        :rtype: SystemModel
        """
        return self.solution_method.model

    @property
    def problem(self) -> OptimalControlProblem:
        """
        :rtype: OptimalControlProblem
        """
        return self.solution_method.problem

    @property
    def degree(self):
        return self.solution_method.degree

    @property
    def degree_control(self):
        return self.solution_method.degree_control

    @property
    def finite_elements(self):
        return self.solution_method.finite_elements

    @property
    def delta_t(self):
        return self.solution_method.delta_t

    @property
    def time_breakpoints(self):
        return self.solution_method.time_breakpoints

    @cached_property
    def time_interpolation_controls(self):
        tau_list = (
            [0.0]
            if self.degree_control == 1
            else self.solution_method.collocation_points(
                self.degree_control, with_zero=False
            )
        )
        return [
            [t + self.solution_method.delta_t * tau for tau in tau_list]
            for t in self.time_breakpoints[:-1]
        ]

    @overload
    def vectorize(self, vector: List[NumericMT]) -> NumericMT:
        ...

    @overload
    def vectorize(self, vector: List[List[NumericMT]]) -> NumericMT:
        ...

    def vectorize(
        self, vector: Union[List[NumericMT], List[List[NumericMT]]]
    ) -> NumericMT:
        #  return vertcat(*[vertcat(*sub_vector) for sub_vector in vector])
        if vector and isinstance(vector[0], list):
            return vertcat(*[self.vectorize(sub_vector) for sub_vector in vector])
        else:
            return vertcat(*vector)

    def _create_function_from_expression(self, name: str, expr: SX):
        """Create a dictionary with range(self.finite_elements) as keys and
        a function of the expression in each finite element.
        The function argument is model.all_sym

        :param str name: function name
        :param expr: symbolic expression
        :return: dictionary with functions
        :rtype: dict
        """
        function_dict = {}
        for el in range(self.finite_elements):
            expr_el = convert_expr_from_tau_to_time(
                expr,
                self.model.t,
                self.model.tau,
                self.time_breakpoints[el],
                self.time_breakpoints[el + 1],
            )
            f_expr_el = Function(name + "_" + str(el), self.model.all_sym, [expr_el])
            function_dict[el] = f_expr_el
        return function_dict

    def discretize(
        self,
        x_0: MX = None,
        p: MX = None,
        theta: Dict[int, MX] = None,
        last_u: MX = None,
    ) -> AbstractOptimizationProblem:
        """Discretize the OCP, returning a Optimization Problem

        :param x_0: initial condition
        :param p: parameters
        :param theta: theta parameters
        :param last_u: last applied control
        :return:
        """
        raise NotImplementedError

    def _create_nlp_symbolic_variables(
        self, nlp: AbstractOptimizationProblem
    ) -> Tuple[MX, List[List[MX]], List[List[MX]], List[List[MX]], MX, MX, List[MX]]:
        """
        Create the symbolic variables that will be used by the NLP problem
        :rtype: (DM, List[List[DM]], List(List(DM)), List(DM), DM, DM, DM)
        """
        raise NotImplementedError

    def get_system_at_given_times(
        self,
        x,
        y,
        u,
        time_dict=None,
        p=None,
        theta=None,
        functions=None,
        start_at_t_0=False,
    ):
        raise NotImplementedError

    def set_data_to_optimization_result_from_raw_data(
        self, optimization_result: OptimizationResult, raw_solution_dict: Dict[str, DM]
    ):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """
        raise NotImplementedError

    def unpack_decision_variables(
        self, decision_variables: MX, all_subinterval: bool = True
    ) -> Tuple[List[List[MX]], List[List[MX]], List[List[MX]], MX, MX, List[MX]]:
        """Return a structured data from the decision variables vector

        Returns:
        (x_data, y_data, u_data, p_opt, eta)

        :param decision_variables: DM
        :return: tuple
        """

        raise NotImplementedError

    def create_initial_guess(self, p=None, theta=None):
        """Create an initial guess for the optimal control problem using problem.x_0, problem.y_guess, problem.u_guess,
        and a given p and theta (for p_opt and theta_opt) if they are given.
        If y_guess or u_guess are None the initial guess uses a vector of zeros of appropriate size.
        If no p or theta is given, an vector of zeros o appropriate size is used.

        :param p: Optimization parameters
        :param theta: Optimization theta
        :return:
        """
        raise NotImplementedError

    def create_initial_guess_with_simulation(
        self,
        u=None,
        p=None,
        theta=None,
        model=None,
        in_transform=None,
        out_transform=None,
    ):
        """Create an initial guess for the optimal control problem using by simulating
        with a given control u, and a given p and theta (for p_opt and theta_opt) if
        they are given.
        If no u is given the value of problem.u_guess is used, or problem.last_u,
        then a vector of zeros of appropriate size is used.
        If no p or theta is given, an vector of zeros of appropriate size is used.

        :param u:
        :param p: Optimization parameters
        :param theta: Optimization theta
        :param model: Model used for simulating, if None is given use problem.model
        :param in_transform: a function that receives (x_0, t_0, t_f, p, y_0) in the form of the
            problem and returns them in the form of the simulation model.
            Does not apply transforms if None
        :param out_transform: a function that receives a CasADi integration result and returns
            an equivalent dict
        :return:
        """
        raise NotImplementedError
