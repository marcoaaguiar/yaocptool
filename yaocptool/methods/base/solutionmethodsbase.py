from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import warnings
from functools import cached_property
from yaocptool.modelling.system_model import SystemModel

from casadi import SX, MX, vertcat, collocation_points, vec, DM, reshape

from yaocptool import config, create_constant_theta, find_variables_indices_in_vector
from yaocptool.methods import SolutionMethodInterface
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.classic.collocationscheme import CollocationScheme
from yaocptool.methods.classic.multipleshooting import MultipleShootingScheme
from yaocptool.modelling import OptimalControlProblem
from yaocptool.optimization.abstract_optimization_problem import (
    AbstractOptimizationProblem,
)

if TYPE_CHECKING:
    from casadi import FunctionCallArgT, NumericMT, SymbolicMT
else:
    FunctionCallArgT = "FunctionCallArgT"
    NumericMT = "NumericMT"
    SymbolicMT = "SymbolicMT"


class SolutionMethodsBase(SolutionMethodInterface):
    degree: int = 3
    degree_control: int = 1
    finite_elements: int = 10

    def __init__(self, problem: OptimalControlProblem, **kwargs):
        """
        :param OptimalControlProblem problem:
        :param str integrator_type: str
        :param str solution_method: str
        :param int degree: discretization polynomial degree
        :param int degree_control:
        :param str discretization_scheme: ('multiple-shooting' | 'collocation')
        :param str initial_guess_heuristic: 'simulation' or 'problem_info'
        :param bool last_control_as_parameter: Default: False, if set to True, the last control will be an
            parameter for the NLP generated from the OCP. This is useful for MPCs, where the initial condition changes
            every iteration.
        """
        self.opt_problem: Optional[AbstractOptimizationProblem] = None
        self.problem = problem
        self.solution_class = ""
        self.prepared = False

        # Options
        self.integrator_type = "implicit"
        self.discretization_scheme = "collocation"
        self.initial_condition_as_parameter = True
        self.nlpsol_opts: Dict[str, Any] = {}
        self.initial_guess_heuristic = "simulation"  # 'problem_info'
        self.last_control_as_parameter = False
        self.initial_guess_model = None
        self.initial_guess_in_transform = None
        self.initial_guess_out_transform = None

        # Internal variables
        self.parametrized_control = False

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        for k in config.SOLVER_OPTIONS["nlpsol_options"]:
            if k not in self.nlpsol_opts:
                self.nlpsol_opts[k] = config.SOLVER_OPTIONS["nlpsol_options"][k]

        if self.problem.last_u is not None:
            self.last_control_as_parameter = True

        if self.discretization_scheme == "multiple-shooting":
            self.discretizer = MultipleShootingScheme(self)
        elif self.discretization_scheme == "collocation":
            self.discretizer = CollocationScheme(self)
        else:
            raise Exception(
                "Discretization scheme not recognized: '{}'. Available options: 'multiple-shooting'"
                " and 'collocation'".format(self.discretization_scheme)
            )

    @property
    def model(self) -> SystemModel:
        return self.problem.model

    @property
    def delta_t(self) -> float:
        return (self.problem.t_f - self.problem.t_0) / self.finite_elements

    @cached_property
    def time_breakpoints(self) -> List[float]:
        return [self.delta_t * k for k in range(self.finite_elements + 1)]

    @property
    def time_interpolation_controls(self) -> List[List[float]]:
        tau_list = (
            [0.0]
            if self.degree_control == 1
            else self.collocation_points(self.degree_control, with_zero=False)
        )
        return [
            [t + self.delta_t * tau for tau in tau_list]
            for t in self.time_breakpoints[:-1]
        ]

    @staticmethod
    def collocation_points(degree, cp="radau", with_zero=False) -> List[float]:
        if with_zero:
            return [0.0] + collocation_points(degree, cp)  # All collocation time points
        else:
            return collocation_points(degree, cp)  # All collocation time points

    def _create_lagrangian_polynomial_basis(
        self, degree: int, starting_index: int = 0, tau: Optional[SX] = None
    ) -> Tuple[SX, List[SX]]:
        if tau is None:
            tau = self.model.tau  # symbolic variable

        tau_root = self.collocation_points(
            degree, with_zero=True
        )  # All collocation time points

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        l_list: List[SX] = []
        for j in range(starting_index, degree + 1):
            ell = SX(1)
            for j2 in range(starting_index, degree + 1):
                if j2 != j:
                    ell *= (tau - tau_root[j2]) / (tau_root[j] - tau_root[j2])
            l_list.append(ell)

        return tau, l_list

    def create_variable_polynomial_approximation(
        self,
        size: int,
        degree: int,
        name: Union[str, List[str]] = "var_appr",
        tau: SX = None,
        point_at_t0: bool = False,
    ) -> Tuple[SX, SX]:
        if not isinstance(name, list):
            name = [name + "_" + str(i) for i in range(size)]

        if tau is None:
            tau = self.model.tau  # Collocation point

        if degree == 1:
            if size > 0:
                points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
            else:
                points = SX.sym("empty_sx", size, degree)
            par = vec(points)
            u_pol = points
        else:
            if point_at_t0:
                if size > 0:
                    points = vertcat(
                        *[SX.sym(name[s], 1, degree + 1) for s in range(size)]
                    )
                else:
                    points = SX.sym("empty_sx", size, degree)
                tau, ell_list = self._create_lagrangian_polynomial_basis(
                    degree, starting_index=0, tau=tau
                )
                u_pol = SX(sum(ell_list[j] * points[:, j] for j in range(degree + 1)))
            else:
                if size > 0:
                    points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
                else:
                    points = SX.sym("empty_sx", size, degree)
                tau, ell_list = self._create_lagrangian_polynomial_basis(
                    degree, starting_index=1, tau=tau
                )
                u_pol = SX(sum(ell_list[j] * points[:, j] for j in range(degree)))
            par = vec(points)

        return u_pol, par

    def create_control_approximation(self) -> SX:
        """Parametrize the control variable, accordingly to the 'degree_control' attribute.
        If degree_control == 1, then a piecewise constant control will be used (most common).
        If degree_control > 1, then a piecewise polynomial approximation will be used with order 'degree_control'.

        :return:
        """
        degree = self.degree_control
        if not self.parametrized_control:
            if type(degree) == dict:
                raise Exception("Not implemented")
            else:
                u_pol, self.model.u_par = self.create_variable_polynomial_approximation(
                    self.model.n_u, degree, name=self.model.u_names
                )
            self.model.u_expr = u_pol
        else:
            u_pol = self.model.u_expr

        return u_pol

    def unvec(self, vector: NumericMT, degree: int = None) -> NumericMT:
        """
        Unvectorize 'vector' a vectorized matrix, assuming that it was a matrix with 'degree' number of columns
        :type vector: DM a vector (flattened matrix)
        :type degree: int
        """
        if degree is None:
            degree = self.degree
        n_lines = vector.numel() // degree

        return reshape(vector, n_lines, degree)

    # ==============================================================================
    # SOLVE
    # ==============================================================================

    def prepare(self):
        """
        Pre solve checks and problem transformations
        """
        if self.prepared:
            return

        self.problem.pre_solve_check()

    def create_optimization_problem(self):
        if not self.prepared:
            self.prepare()
            self.prepared = True

        has_parameters = (
            self.model.n_p + self.model.n_theta > 0
            or self.initial_condition_as_parameter
            or self.problem.last_u is not None
        )

        # parameters MX
        p_mx = MX.sym("mx_p", self.model.n_p)

        # theta MX
        theta_mx = MX.sym("mx_theta_", self.model.n_theta, self.finite_elements)
        theta = {i: vec(theta_mx[:, i]) for i in range(self.finite_elements)}

        # initial cond MX
        p_mx_x_0 = MX.sym("mx_x_0_p", self.model.n_x)

        # last control MX
        if self.last_control_as_parameter:
            p_last_u = MX.sym("mx_last_u", self.model.n_u)
        else:
            p_last_u = MX()

        all_mx = vertcat(p_mx, vec(theta_mx), p_mx_x_0, p_last_u)

        args = {"p": p_mx, "x_0": p_mx_x_0, "theta": theta, "last_u": p_last_u}

        # Discretize the problem
        opt_problem = self.discretizer.discretize(
            p=p_mx, x_0=p_mx_x_0, theta=theta, last_u=p_last_u
        )

        if has_parameters:
            opt_problem.include_parameter(all_mx)

        opt_problem.solver_options = self.nlpsol_opts

        self.opt_problem = opt_problem

    def call_solver(
        self,
        initial_guess=None,
        p=None,
        theta=None,
        x_0=None,
        last_u=None,
        initial_guess_dict=None,
    ):
        if self.opt_problem is None:
            self.create_optimization_problem()

        # initial conditions
        args, p, theta, x_0, last_u = self._get_solver_call_args(
            x_0, p, theta, last_u, initial_guess_dict, initial_guess
        )

        sol = self.opt_problem.solve(**args)
        return sol, p, theta, x_0, last_u

    def _get_solver_call_args(
        self, x_0, p, theta, last_u, initial_guess_dict, initial_guess
    ):
        # initial conditions
        if x_0 is None:
            x_0 = self.problem.x_0
        if isinstance(x_0, list):
            x_0 = vertcat(x_0)

        if vertcat(x_0).numel() != self.model.n_x:
            raise Exception(
                "Size of given x_0 (or obtained from problem.x_0) is different from model.n_x, "
                "x_0.numel() = {}, model.n_x = {}".format(
                    vertcat(x_0).numel(), self.model.n_x
                )
            )

        # parameters
        if p is None:
            if self.problem.n_p_opt == self.model.n_p:
                p = DM.zeros(self.problem.n_p_opt)
            elif self.problem.model.n_p > 0:
                raise Exception(
                    "A parameter 'p' of size {} should be given".format(
                        self.problem.model.n_p
                    )
                )

        if isinstance(p, list):
            p = DM(p)

        # theta
        if theta is None:
            if self.problem.n_theta_opt == self.model.n_theta:
                theta = create_constant_theta(
                    0, self.problem.n_theta_opt, self.finite_elements
                )
            elif self.problem.model.n_theta > 0:
                raise Exception(
                    "A parameter 'theta' of size {} should be given".format(
                        self.problem.model.n_theta
                    )
                )

        # Prepare NLP parameter vector
        theta_vector, par_x_0, par_last_u = [], [], []
        if theta is not None:
            theta_vector = vertcat(*[theta[i] for i in range(self.finite_elements)])

        if self.initial_condition_as_parameter:
            par_x_0 = x_0

        # last control
        if not self.last_control_as_parameter and last_u is not None:
            warnings.warn(
                "solution_method.last_control_as_parameter is False, but last_u was passed."
                "last_u will be ignored."
            )

        if last_u is not None:
            if isinstance(last_u, list):
                last_u = vertcat(*last_u)
        elif self.problem.last_u is not None:
            last_u = self.problem.last_u
        elif self.last_control_as_parameter:
            raise Exception(
                'last_control_as_parameter is True, but no "last_u" was passed'
                'and the "ocp.last_u" is None.'
            )

        if self.last_control_as_parameter:
            par_last_u = last_u

        par = vertcat(p, theta_vector, par_x_0, par_last_u)

        if initial_guess_dict is None:
            if initial_guess is None:
                if self.initial_guess_heuristic == "simulation":
                    initial_guess = (
                        self.discretizer.create_initial_guess_with_simulation(
                            p=p,
                            theta=theta,
                            model=self.initial_guess_model,
                            in_transform=self.initial_guess_in_transform,
                            out_transform=self.initial_guess_out_transform,
                        )
                    )
                elif self.initial_guess_heuristic == "problem_info":
                    initial_guess = self.discretizer.create_initial_guess(p, theta)
                else:
                    raise ValueError(
                        "initial_guess_heuristic did not recognized, available options:"
                        '"simulation" and "problem_info". Given: {}'.format(
                            self.initial_guess_heuristic
                        )
                    )
            args = dict(initial_guess=initial_guess, p=par)
        elif isinstance(initial_guess, dict):
            initial_guess = vertcat(
                vec(initial_guess[key])
                for key in ["x", "y", "u", "eta", "p_opt", "theta_opt"]
            )
            args = dict(initial_guess=initial_guess, p=par)
        else:
            args = dict(
                initial_guess=initial_guess_dict["x"],
                p=par,
                lam_x=initial_guess_dict["lam_x"],
                lam_g=initial_guess_dict["lam_g"],
            )
        return args, p, theta, x_0, last_u

    def solve(
        self,
        initial_guess=None,
        p=None,
        theta=None,
        x_0=None,
        last_u=None,
        initial_guess_dict=None,
    ) -> OptimizationResult:
        """

        :param initial_guess: Initial guess
        :param p: Parameters values
        :param theta: Theta values
        :param x_0: Initial condition value
        :param last_u: Last control value
        :param initial_guess_dict: Initial guess as dict
        :rtype: OptimizationResult
        """
        if isinstance(p, (int, float)):
            p = DM(p)

        raw_solution_dict, p, theta, x_0, last_u = self.call_solver(
            initial_guess=initial_guess,
            p=p,
            theta=theta,
            x_0=x_0,
            last_u=last_u,
            initial_guess_dict=initial_guess_dict,
        )

        return self.create_optimization_result(raw_solution_dict, p, theta, x_0)

    def mp_solve(
        self,
        initial_guess=None,
        p=None,
        theta=None,
        x_0=None,
        last_u=None,
        initial_guess_dict=None,
    ) -> OptimizationResult:
        if isinstance(p, (int, float)):
            p = DM(p)

        # initial conditions
        args, p, theta, x_0, last_u = self._get_solver_call_args(
            x_0, p, theta, last_u, initial_guess_dict, initial_guess
        )

        sol = self.opt_problem.mp_solve(**args)
        return sol, p, theta, x_0, last_u

        return self.create_optimization_result(raw_solution_dict, p, theta, x_0)

    def create_optimization_result(self, raw_solution_dict, p, theta, x_0):
        optimization_result = OptimizationResult()

        # From the solution_method
        for attr in [
            "finite_elements",
            "degree",
            "degree_control",
            "time_breakpoints",
            "discretization_scheme",
        ]:
            attr_value = getattr(self, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.method_name = self.__class__.__name__

        # Initial condition, theta, and parameters used
        optimization_result.x_0 = x_0
        optimization_result.theta = theta
        optimization_result.p = p

        # From the problem
        for attr in ["t_0", "t_f"]:
            attr_value = getattr(self.problem, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.problem_name = self.problem.name

        # From model
        optimization_result.x_names = [
            self.model.x[i].name() for i in range(self.model.n_x)
        ]
        optimization_result.y_names = [
            self.model.y[i].name() for i in range(self.model.n_y)
        ]
        optimization_result.u_names = [
            self.model.u[i].name() for i in range(self.model.n_u)
        ]
        optimization_result.theta_opt_names = [
            self.problem.theta_opt[i].name() for i in range(self.problem.n_theta_opt)
        ]

        self.discretizer.set_data_to_optimization_result_from_raw_data(
            optimization_result, raw_solution_dict
        )

        if self.problem.x_cost is not None:
            x_c_index = find_variables_indices_in_vector(
                self.problem.x_cost, self.model.x
            )
            optimization_result.x_c_final = optimization_result.x_data["values"][-1][
                -1
            ][x_c_index]

        return optimization_result
