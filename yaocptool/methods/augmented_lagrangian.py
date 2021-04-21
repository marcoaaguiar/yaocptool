# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Type, Union

from casadi import (
    DM,
    MX,
    SX,
    Function,
    collocation_points,
    diag,
    dot,
    fabs,
    fmin,
    horzcat,
    inf,
    mmax,
    substitute,
    vec,
    vertcat,
    vertsplit,
)

from yaocptool import (
    create_constant_theta,
    find_variables_indices_in_vector,
    join_thetas,
    remove_variables_from_vector_by_indices,
)
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.base.solutionmethodinterface import SolutionMethodInterface
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
from yaocptool.modelling.ocp import OptimalControlProblem
from yaocptool.modelling.system_model import SystemModel
from yaocptool.optimization.abstract_optimization_problem import (
    ExtendedOptiResultDictType,
    OptiResultDictType,
)
from yaocptool.util.util import create_polynomial_approximation

LOGGER = logging.getLogger(__name__)


@dataclass
class AugmentedLagrangianOptions:
    degree: int = 3
    finite_elements: int = 20

    max_iter: int = 20
    tol: float = 1e-6
    mu_0: DM = DM(1.0)
    mu_max: DM = DM(1e4)
    beta: DM = DM(4.0)

    only_update_if_improve: bool = False
    #  mu_update_rule: str = "simple"
    mu_update_rule: Union[
        Literal["error_dependent"], Literal["simple"]
    ] = "error_dependent"
    mu_min_error_decrease: float = 0.5

    verbose: int = 2
    debug_skip_parametrize: bool = False
    debug_skip_initialize: bool = False
    debug_skip_compute_nu_and_error: bool = False
    debug_skip_update_nu: bool = False
    debug_skip_update_mu: bool = False

    def __post_init__(self):
        self.mu_0 = DM(self.mu_0)
        self.mu_max = DM(self.mu_max)
        self.beta = DM(self.beta)


class OptionsOverride(type):
    """
    Reroute SolutionMethod options to its self.options
    """

    def __new__(cls, clsname, bases, attrs):
        def generate_getter_setter(attr):
            def _getter(self):
                LOGGER.error(f"getting {attr} = {getattr(self.options, attr)}")
                return getattr(self.options, attr)

            def _setter(self, val: Any):
                return setattr(self.options, attr, val)

            return _getter, _setter

        for attr in dir(AugmentedLagrangianOptions):
            if attr.startswith("__"):
                continue
            attrs[attr] = property(*generate_getter_setter(attr))

        return super().__new__(cls, clsname, bases, attrs)


class AugmentedLagrangian(SolutionMethodInterface, metaclass=OptionsOverride):
    def __init__(
        self,
        problem: OptimalControlProblem,
        ocp_solver_class: Type[SolutionMethodsBase],
        solver_options: Optional[Dict[str, Any]] = None,
        options: Union[AugmentedLagrangianOptions, Dict[str, Any]] = None,
        nu: Optional[Dict[int, DM]] = None,
        relax_algebraic_index: List[int] = None,
        relax_algebraic_var_index: List[int] = None,
        relax_time_equality_index: List[int] = None,
        relax_time_equality_var_index: List[int] = None,
        relax_state_bounds: bool = False,
        no_update_after_solving: bool = True,
        **kwargs,
    ):
        """
            Augmented Lagrange Method (Aguiar 2016)

        :param problem: Optimal Control Problem
        :param ocp_solver_class: Class of Solution Method (Direct/Indirect Method)
        :param solver_options: Options for the Solution Method class given
        :param relax_algebraic_index: Index for the algebraic equations that will be relaxed,
            if not given all the algebraic equations will be relaxed
        :param relax_algebraic_var_index: Index for the algebraic variables that will be relaxed,
            if not given it will be assumed the same as the 'relax_algebraic_index'
        :param bool relax_state_bounds: This relax the states bounds and put then in the objective,
            via an algebraic variable
        :param kwargs:
        """
        if solver_options is None:
            solver_options = {}
        if options is None:
            options = {}

        self.problem = problem

        self.n_relax = 0
        self.mu_sym = SX()
        self.nu_sym = SX()
        self.nu_par = SX()
        self.nu_pol = SX()

        #  for key, val in [*kwargs.items()]:
        #      print(key, val)
        #      if hasattr(AugmentedLagrangianOptions, key):
        #          if isinstance(options, dict):
        #              options[key] = kwargs.pop(key)
        #          else:
        #              setattr(options, key, val)
        #          warnings.warn(
        #              f"Pass options as a dict in options keyword argument, {key}={val}",
        #              Warning,
        #          )

        self.options = (
            AugmentedLagrangianOptions(**options)
            if isinstance(options, dict)
            else options
        )
        LOGGER.setLevel(
            {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}[
                self.options.verbose
            ]
        )

        self.nu_tilde = None
        self.last_violation_error = None
        self.new_nu_func = None
        self.opt_problem = None

        self.last_solution = ()

        self.solver = None
        self.ocp_solver: Optional[SolutionMethodsBase] = None
        self.solver_initialized = False

        self.relax_algebraic_index = (
            relax_algebraic_index
            if relax_algebraic_index is not None
            else list(range(self.model.n_y))
        )
        self.relax_algebraic_var_index = (
            relax_algebraic_var_index
            if relax_algebraic_var_index is not None
            else self.relax_algebraic_index
        )
        self.relax_time_equality_index = relax_time_equality_index or []
        self.relax_time_equality_var_index = (
            relax_time_equality_var_index or self.relax_time_equality_index
        )
        self.relax_state_bounds = relax_state_bounds
        self.no_update_after_solving = no_update_after_solving

        self.relaxed_alg = SX()
        self.relaxed_eq = SX()

        self.mu = DM()

        # RELAXATION

        if self.model.alg[self.relax_algebraic_index].numel() > 0:
            self._relax_algebraic_equations()

        if self.model.alg[self.relax_time_equality_index].numel() > 0:
            self._relax_time_equalities()

        if self.relax_state_bounds:
            self._relax_states_constraints()

        if not self.options.debug_skip_initialize:
            if not self.options.debug_skip_parametrize:
                self._parametrize_nu()

            self.nu = nu or self.create_nu_initial_guess()
            self.alg_violation = create_constant_theta(
                constant=0,
                dimension=len(self.relax_algebraic_index) * self.options.degree,
                finite_elements=self.options.finite_elements,
            )
            self.eq_violation = create_constant_theta(
                constant=0,
                dimension=len(self.relax_time_equality_index) * self.options.degree,
                finite_elements=self.options.finite_elements,
            )

        #  make sure that the ocp_solver and the augmented_lagrangian has the same options
        for attr in ["degree", "finite_elements"]:
            if (
                attr in solver_options
                and attr in kwargs
                and solver_options[attr] != kwargs[attr]
            ):
                exc_mess = "Trying to pass attribute '{}' for '{}' and '{}' that are not equal: {} != {}"
                raise ValueError(
                    exc_mess.format(
                        attr,
                        self.__class__.__name__,
                        ocp_solver_class.__name__,
                        kwargs[attr],
                        solver_options[attr],
                    )
                )
            elif attr in solver_options:
                setattr(self.options, attr, solver_options[attr])
            else:
                solver_options[attr] = getattr(self.options, attr)

        #  solver_options["integrator_type"] = self.integrator_type

        # Initialize OCP solver
        self.ocp_solver = ocp_solver_class(self.problem, **solver_options)

    # region # PROPERTY
    @property
    def model(self) -> SystemModel:
        return self.problem.model

    @property
    def discretizer(self) -> DiscretizationSchemeBase:
        return self.ocp_solver.discretizer

    @property
    def time_interpolation_nu(self) -> List[List[float]]:
        col_points = collocation_points(self.options.degree, "radau")
        return [
            [
                self.ocp_solver.time_breakpoints[el]
                + self.ocp_solver.delta_t * col_points[j]
                for j in range(self.options.degree)
            ]
            for el in range(self.options.finite_elements)
        ]

    # endregion

    # ==============================================================================
    # region RELAX

    def _include_relax_in_objective(self, relaxed_eqs: SX):
        n_alg_relax = relaxed_eqs.numel()

        self.n_relax += n_alg_relax

        # create a symbolic nu
        nu_alg = SX.sym("AL_nu_alg", n_alg_relax)
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        mu_sym = self._create_mu_variable(n_alg_relax)

        # include the penalization term in the objective
        self.problem.L += (
            nu_alg.T @ relaxed_eqs
            + 1 / 2.0 * relaxed_eqs.T @ diag(mu_sym) @ relaxed_eqs
        )

    def _relax_algebraic_equations(self):
        # get the equations to relax
        alg_relax = self.model.alg[self.relax_algebraic_index]

        # save the relaxed algebraic equations for computing the update later
        self.relaxed_alg = vertcat(self.relaxed_alg, alg_relax)

        # include in the objective function
        self._include_relax_in_objective(alg_relax)

        # include the relaxed y_sym as controls
        u_guess = (
            self.problem.y_guess[self.relax_algebraic_index]
            if self.problem.y_guess is not None
            else None
        )

        self.problem.include_control(
            self.model.y[self.relax_algebraic_var_index],
            u_max=self.problem.y_max[self.relax_algebraic_var_index],
            u_min=self.problem.y_min[self.relax_algebraic_var_index],
            u_guess=u_guess,
        )
        self.problem.remove_algebraic(
            self.model.y[self.relax_algebraic_var_index], alg_relax
        )

    def _relax_time_equalities(self):
        # get the equations to relax
        eq_relax = self.problem.g_eq[self.relax_time_equality_index]

        # save the relaxed algebraic equations for computing the update later
        self.relaxed_eq = vertcat(self.relaxed_eq, eq_relax)

        # include in the objective function
        self._include_relax_in_objective(eq_relax)

        # Remove equality
        self.problem.g_eq = remove_variables_from_vector_by_indices(
            self.relax_time_equality_index, self.problem.g_eq
        )

        for ind in sorted(self.relax_time_equality_index, reverse=True):
            self.problem.time_g_eq.pop(ind)

    def _relax_states_constraints(self):
        for i in range(self.model.n_x):
            if self.problem.x_max[i] != inf or self.problem.x_min[i] != -inf:
                y_x = SX.sym("y_x_" + str(i))
                nu_y_x = SX.sym("nu_y_x_" + str(i))

                self.nu_sym = vertcat(self.nu_sym, nu_y_x)
                mu_sym = self._create_mu_variable()

                new_alg = y_x - self.model.x[i]
                self.problem.L += dot(nu_y_x.T, new_alg) + mu_sym / 2.0 * dot(
                    new_alg.T, new_alg
                )

                self.relaxed_alg = vertcat(self.relaxed_alg, new_alg)
                self.problem.include_control(
                    y_x, u_min=self.problem.x_min[i], u_max=self.problem.x_max[i]
                )
                self.problem.x_max[i] = inf
                self.problem.x_min[i] = -inf
                self.n_relax += 1

    def _relax_inequalities(self):
        raise NotImplementedError()

    def _parametrize_nu(self):
        nu_pol, nu_par = create_polynomial_approximation(
            self.problem.model.tau, self.n_relax, self.options.degree, "nu"
        )

        self.nu_pol = vertcat(self.nu_pol, nu_pol)
        self.nu_par = vertcat(self.nu_par, nu_par)

        self.problem.replace_variable(self.nu_sym, nu_pol)
        self.problem.model.include_theta(vec(nu_par))

        return nu_pol, nu_par

    def _create_mu_variable(self, size: int = 1) -> SX:
        new_mu = self.problem.create_parameter(f"mu_{self.mu_sym.numel()}", size)
        self.mu_sym = vertcat(self.mu_sym, new_mu)
        self.mu = vertcat(self.mu, *([self.options.mu_0] * size))
        return new_mu

    def create_nu_initial_guess(self):
        return create_constant_theta(
            constant=0,
            dimension=self.n_relax * self.options.degree,
            finite_elements=self.options.finite_elements,
        )

    def _create_nu_update_func(self):
        v = self.ocp_solver.opt_problem.x

        (
            x_var,
            y_var,
            u_var,
            _,
            p_opt,
            theta_opt,
        ) = self.ocp_solver.discretizer.unpack_decision_variables(v)
        par = MX.sym("par", self.model.n_p)
        theta = {
            i: vec(MX.sym("theta_" + repr(i), self.model.n_theta))
            for i in range(self.options.finite_elements)
        }
        theta_var = vertcat(*[theta[i] for i in range(self.options.finite_elements)])

        time_dict = {i: {} for i in range(self.options.finite_elements)}

        for el in range(self.options.finite_elements):
            time_dict[el]["t_0"] = self.ocp_solver.time_breakpoints[el]
            time_dict[el]["t_f"] = self.ocp_solver.time_breakpoints[el + 1]
            time_dict[el]["f_nu"] = self.time_interpolation_nu[el]
            time_dict[el]["f_relax_alg"] = self.time_interpolation_nu[el]
            time_dict[el]["f_relax_eq"] = self.time_interpolation_nu[el]

        functions = defaultdict(dict)
        for el in range(self.options.finite_elements):
            func_rel_alg = self.model.convert_expr_from_tau_to_time(
                substitute(self.relaxed_alg, self.model.u, self.model.u_expr),
                self.ocp_solver.time_breakpoints[el],
                self.ocp_solver.time_breakpoints[el + 1],
            )
            func_rel_eq = self.model.convert_expr_from_tau_to_time(
                substitute(self.relaxed_eq, self.model.u, self.model.u_expr),
                self.ocp_solver.time_breakpoints[el],
                self.ocp_solver.time_breakpoints[el + 1],
            )
            nu_time_dependent = self.model.convert_expr_from_tau_to_time(
                self.nu_pol,
                self.ocp_solver.time_breakpoints[el],
                self.ocp_solver.time_breakpoints[el + 1],
            )

            f_nu = Function("f_nu", self.model.all_sym, [nu_time_dependent])
            f_relax_alg = Function("f_relax_alg", self.model.all_sym, [func_rel_alg])
            f_relax_eq = Function("f_relax_eq", self.model.all_sym, [func_rel_eq])

            functions["f_nu"][el] = f_nu
            functions["f_relax_alg"][el] = f_relax_alg
            functions["f_relax_eq"][el] = f_relax_eq

        # get values (symbolically)
        results = self.ocp_solver.discretizer.get_system_at_given_times(
            x_var, y_var, u_var, time_dict, p=par, theta=theta, functions=functions
        )

        # compute new nu
        mu_mx = par[find_variables_indices_in_vector(self.mu_sym, self.model.p)]
        new_nu = []
        rel_alg = []
        rel_eq = []
        for el in range(self.options.finite_elements):
            new_nu_k = []
            rel_alg_k = []
            rel_eq_k = []
            for j in range(self.options.degree):
                nu_kj = results[el]["f_nu"][j]
                rel_alg_kj = results[el]["f_relax_alg"][j]
                rel_eq_kj = results[el]["f_relax_eq"][j]
                rel_kj = vertcat(rel_alg_kj, rel_eq_kj)

                new_nu_k = horzcat(new_nu_k, nu_kj + mu_mx * rel_kj)
                rel_alg_k = horzcat(rel_alg_k, rel_alg_kj)
                rel_eq_k = horzcat(rel_eq_k, rel_eq_kj)
            new_nu.append(new_nu_k)
            rel_alg.append(rel_alg_k)
            rel_eq.append(rel_eq_k)

        output = [horzcat(*new_nu), horzcat(*rel_alg), horzcat(*rel_eq)]

        return Function("nu_update_function", [v, par, theta_var], output)

    @staticmethod
    def join_nu_to_theta(theta, nu):
        if theta is not None:
            return join_thetas(theta, nu)
        else:
            return nu

    def create_optimization_problem(self):
        self.ocp_solver.create_optimization_problem()
        self.opt_problem = self.ocp_solver.opt_problem

    def solve(
        self,
        initial_guess: Optional[DM] = None,
        p: Optional[Union[DM, List[float]]] = None,
        theta: Optional[Dict[int, DM]] = None,
        x_0: Optional[Union[DM, List[float]]] = None,
        last_u: Optional[DM] = None,
        initial_guess_dict: Optional[OptiResultDictType] = None,
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
        if isinstance(p, (int, float, list)):
            p = DM(p)
        if isinstance(x_0, (int, float, list)):
            x_0 = DM(x_0)

        raw_solution_dict, p, theta, x_0, last_u = self.call_solver(
            initial_guess=initial_guess,
            p=p,
            theta=theta,
            x_0=x_0,
            last_u=last_u,
            initial_guess_dict=initial_guess_dict,
        )

        return self.create_optimization_result(raw_solution_dict, p, theta, x_0)

    def call_solver(
        self,
        initial_guess: DM = None,
        p: Optional[DM] = None,
        theta: Optional[Dict[int, DM]] = None,
        x_0: Optional[DM] = None,
        last_u: Optional[DM] = None,
        initial_guess_dict: Optional[OptiResultDictType] = None,
    ):
        #  if self.opt_problem is None:
        #  self.create_optimization_problem()

        # initialize variables
        t_0 = time.time()
        n_it = raw_solution_dict = p_k = theta_k = x_0 = last_u = error = None
        for n_it in range(self.options.max_iter):
            t_1 = time.time()
            if self.nu_tilde is not None:
                self.nu = self.nu_tilde

            theta_k = self.join_nu_to_theta(theta, self.nu)
            p_k = vertcat(p, self.mu) if p is not None else self.mu

            # solve the optimization problem
            optimization_result = self.ocp_solver.solve(
                initial_guess=initial_guess,
                p=p_k,
                theta=theta_k,
                x_0=x_0,
                last_u=last_u,
                initial_guess_dict=initial_guess_dict,
            )
            raw_solution_dict = optimization_result.raw_solution_dict
            initial_guess_dict = raw_solution_dict

            # update parameters
            error = (
                self.update_parameters(p, theta, raw_solution_dict)
                if not self.no_update_after_solving
                else DM(0)
            )

            if n_it == 0:
                LOGGER.debug(
                    "{} | {} | {}".format("Iter.", " Viol. Error", "Sol. Time")
                )

            if error is not None:
                LOGGER.debug(
                    "{:>5} | {:e} | {:>9.3f}".format(
                        n_it, float(mmax(error)), time.time() - t_1
                    )
                )
            else:
                LOGGER.debug(
                    "{:>5} | {} | {:>9.3f}".format(
                        n_it, "Not computed", time.time() - t_1
                    )
                )

            # Exit condition: error < tol
            if error is not None and all((error < self.options.tol).nz):
                LOGGER.info(
                    "=== Exiting: {} | Viol. Error: {} | Total time: {} ===".format(
                        "Tolerance met", error, time.time() - t_0
                    )
                )
                return raw_solution_dict, p_k, theta_k, x_0, last_u

        if (
            n_it is None
            or raw_solution_dict is None
            or p_k is None
            or theta_k is None
            or error is None
        ):
            raise ValueError("Solver was never called make sure that max_iter > 0")
        # Exit condition: max_iter
        if n_it == self.options.max_iter:
            LOGGER.info(
                "=== Exiting: {} | Viol. Error: {} | Total time: {} ===".format(
                    "Max iteration reached", error, time.time() - t_0
                )
            )
        return raw_solution_dict, p_k, theta_k, x_0, last_u

    def update_parameters(
        self,
        p: Optional[DM],
        theta: Optional[Dict[int, DM]],
        raw_solution_dict: ExtendedOptiResultDictType,
    ) -> DM:
        theta_k = join_thetas(theta, self.nu)
        p_k = vertcat(p, self.mu) if p is not None else self.mu
        error = self._compute_new_nu_and_error(
            p=p_k, theta=theta_k, raw_solution_dict=raw_solution_dict
        )
        self._update_mu(error, self.last_violation_error)
        self.last_violation_error = error

        return error

    def _update_mu(self, error: DM, last_error: Optional[DM]):
        if self.options.debug_skip_update_mu:
            return
        if self.options.mu_update_rule == "error_dependent" and last_error is not None:
            for ind, (mu, err, last_err) in enumerate(
                zip(self.mu.nz, error.nz, last_error.nz)
            ):
                if err <= self.options.mu_min_error_decrease * last_err:
                    LOGGER.info(
                        f"{self.problem.name} {ind}: same mu {mu}: {err} <= {self.options.mu_min_error_decrease} * {last_err} = {self.options.mu_min_error_decrease * last_err}"
                    )
                    self.mu[ind] = DM(mu)
                else:
                    LOGGER.info(
                        f"{self.problem.name} {ind}: increasing mu {mu}: {err} > {self.options.mu_min_error_decrease} * {last_err} = {self.options.mu_min_error_decrease * last_err}"
                    )
                    self.mu[ind] = DM(fmin(self.options.mu_max, mu * self.options.beta))
        elif self.options.mu_update_rule == "simple":
            self.mu = fmin(self.options.mu_max, self.mu * self.options.beta)
            LOGGER.debug(f"{self.problem.name}: mu {self.mu}")

    def _compute_new_nu_and_error(
        self,
        p: DM,
        theta: Dict[int, DM],
        raw_solution_dict: ExtendedOptiResultDictType,
    ) -> DM:
        if self.new_nu_func is None:
            self.new_nu_func = self._create_nu_update_func()

        errors = DM(0)

        if not self.options.debug_skip_compute_nu_and_error:
            raw_decision_variables = raw_solution_dict["x"]
            theta_vector = vertcat(
                *[theta[i] for i in range(self.options.finite_elements)]
            )
            (new_nu, rel_alg, rel_eq) = self.new_nu_func(
                raw_decision_variables, p, theta_vector
            )

            self.alg_violation = (
                {
                    el: rel_alg[
                        :, el * self.options.degree : (el + 1) * self.options.degree
                    ]
                    for el in range(self.options.finite_elements)
                }
                if rel_alg.numel() > 0
                else create_constant_theta(
                    0, (0, self.options.degree), self.options.finite_elements
                )
            )
            self.eq_violation = (
                {
                    el: rel_eq[
                        :, el * self.options.degree : (el + 1) * self.options.degree
                    ]
                    for el in range(self.options.finite_elements)
                }
                if rel_eq.numel() > 0
                else create_constant_theta(
                    0, (0, self.options.degree), self.options.finite_elements
                )
            )

            errors = DM(
                [mmax(fabs(row)) for row in vertsplit(vertcat(rel_alg, rel_eq), 1)]
            )

            nu_tilde = {
                el: vec(
                    new_nu[:, el * self.options.degree : (el + 1) * self.options.degree]
                )
                for el in range(self.options.finite_elements)
            }

            if not self.options.debug_skip_update_nu:
                if (
                    self.options.only_update_if_improve
                    and any((errors > 0).nz)
                    and self.last_violation_error is not None
                ):
                    for ind, (error, last_error) in enumerate(
                        zip(errors.nz, self.last_violation_error.nz)
                    ):
                        if (error - last_error) < 0:
                            self.last_violation_error[ind] = error
                            for el in self.nu_tilde:
                                self.nu_tilde[el][ind] = nu_tilde[el][ind]
                        else:
                            LOGGER.info("Error is greater than zero")
                else:
                    self.nu_tilde = nu_tilde
        return errors

    def create_optimization_result(
        self,
        raw_solution_dict: ExtendedOptiResultDictType,
        p: DM = None,
        theta: Dict[int, DM] = None,
        x_0: DM = None,
    ) -> OptimizationResult:
        if p is None:
            p = DM()
        if theta is None:
            theta = {}

        optimization_result = self.ocp_solver.create_optimization_result(
            raw_solution_dict, p=p, theta=theta, x_0=x_0
        )

        optimization_result.other_data["nu"]["values"] = [
            [
                self.ocp_solver.unvec(self.nu[el])[:, d]
                for d in range(self.options.degree)
            ]
            for el in range(self.options.finite_elements)
        ]
        optimization_result.other_data["nu"]["time"] = self.time_interpolation_nu

        optimization_result.other_data["alg_violation"]["values"] = [
            [
                self.ocp_solver.unvec(self.alg_violation[el])[:, d]
                for d in range(self.options.degree)
            ]
            for el in range(self.options.finite_elements)
        ]
        optimization_result.other_data["alg_violation"][
            "time"
        ] = self.time_interpolation_nu

        optimization_result.other_data["eq_violation"]["values"] = [
            [
                self.ocp_solver.unvec(self.eq_violation[el])[:, d]
                for d in range(self.options.degree)
            ]
            for el in range(self.options.finite_elements)
        ]
        optimization_result.other_data["eq_violation"][
            "time"
        ] = self.time_interpolation_nu

        optimization_result.statistics["violation_error"] = self.last_violation_error
        return optimization_result
