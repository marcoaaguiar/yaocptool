# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import time
import warnings
from collections import defaultdict

from casadi import SX, inf, vertcat, dot, vec, Function, MX, horzcat, mtimes, repmat, mmax, fabs, substitute

from yaocptool import find_variables_indices_in_vector, join_thetas, create_constant_theta
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


class AugmentedLagrangian(SolutionMethodsBase):
    """
        For a minimization problem in the form
            min f(x,u) = \int L(x,u) dt
            s.t.: \dot{x} = f(x,u),
            g_ineq (x,u) \leq 0

        Transforms the problem in a sequence of solution of the problem
            min f(x,u) = \int L(x,u) -\mu \sum \log(-g_ineq(x,u)) dt
            s.t.: \dot{x} = f(x,u),
    """

    def __init__(self, problem, ocp_solver_class, solver_options=None, **kwargs):
        """

        :param yaocptool.modelling.OptimalControlProblem: Optimal Control Problem
        :param yaocptool.methods.SolutionMethodsBase ocp_solver_class: Class of Solution Method (Direct/Indirect Method)
        :param solver_options: Options for the Solution Method class given
        :param relax_algebraic_index: Index for the algebraic equations that will be relaxed, if not given all the
        algebraic equations will be relaxed
        :param relax_algebraic_var_index: Index for the algebraic variables that will be relaxed, if not given it will
        be assumed the same as the 'relax_algebraic_index'
        :param relax_state_bounds: This relax the states bounds and put then in the objective, via an algebraic variable
        :param kwargs:
        """
        if solver_options is None:
            solver_options = {}

        self.degree = 3
        self.degree_control = 3

        self.n_relax = 0
        self.mu_sym = None
        self.nu_sym = []
        self.nu_par = vertcat([])
        self.nu_pol = vertcat([])

        self.max_iter = 3
        self.mu_0 = 10.
        self.beta = 4.
        self.mu_max = self.mu_0 * self.beta ** 10
        self.nu = None
        self.nu_tilde = None
        self.last_violation_error = -1
        self.alg_violation = None
        self.new_nu_func = None  # type: Function

        self.tol = 1e-4
        self.last_solution = ()

        self.solver = None
        self.ocp_solver = None  # type: SolutionMethodsBase
        self.solver_initialized = False

        self.relax_algebraic_index = None
        self.relax_algebraic_var_index = None
        self.relax_state_bounds = False

        self.relaxed_alg = []

        self._debug_skip_parametrize = False
        self._debug_skip_initialize = False
        self._debug_skip_update_nu = False
        self._debug_skip_update_mu = False

        super(AugmentedLagrangian, self).__init__(problem, **kwargs)

        self.mu = self.mu_0

        # RELAXATION
        self.mu_sym = self.problem.create_parameter('mu')

        if self.relax_algebraic_index is None:
            self.relax_algebraic_index = range(self.model.n_y)
        if self.relax_algebraic_var_index is None:
            self.relax_algebraic_var_index = self.relax_algebraic_index

        self._relax_algebraic_equations()

        if self.relax_state_bounds:
            self._relax_states_constraints()

        if not self._debug_skip_initialize:
            if not self._debug_skip_parametrize:
                self._parametrize_nu()

            if self.nu is None:
                self.nu = self.create_nu_initial_guess()
            if self.alg_violation is None:
                self.alg_violation = create_constant_theta(constant=0, dimension=self.n_relax * self.degree,
                                                           finite_elements=self.finite_elements)

        # make sure that the ocp_solver and the augmented_lagrangian has the same options
        for attr in ['degree', 'finite_elements', 'degree_control', 'integrator_type']:
            if attr in solver_options and attr in kwargs and solver_options[attr] != kwargs[attr]:
                exc_mess = "Trying to pass attribute '{}' for '{}' and '{}' that are not equal: {} != {}"
                raise Exception(exc_mess.format(attr, self.__class__.__name__, ocp_solver_class.__name__, kwargs[attr],
                                                solver_options[attr]))
            elif attr in solver_options:
                setattr(self, attr, solver_options[attr])
            else:
                solver_options[attr] = getattr(self, attr)

        solver_options['integrator_type'] = self.integrator_type

        # Initialize OCP solver
        self.ocp_solver = ocp_solver_class(self.problem, **solver_options)  # type: SolutionMethodsBase

    # region # PROPERTY
    @property
    def model(self):
        return self.problem.model

    @property
    def discretizer(self):
        return self.ocp_solver.discretizer

    @discretizer.setter
    def discretizer(self, value):
        pass

    @property
    def time_interpolation_nu(self):
        col_points = self.collocation_points(self.degree, with_zero=False)
        return [[self.time_breakpoints[el] + self.delta_t * col_points[j] for j in
                 range(self.degree)] for el in range(self.finite_elements)]

    # endregion

    # ==============================================================================
    # region RELAX

    def _relax_algebraic_equations(self):
        # get the equations to relax
        alg_relax = self.model.alg[self.relax_algebraic_index]
        n_alg_relax = alg_relax.numel()

        self.n_relax += n_alg_relax

        # create a symbolic nu
        nu_alg = SX.sym('AL_nu_alg', n_alg_relax)
        self.nu_sym = vertcat(self.nu_sym, nu_alg)

        # save the relaxed algebraic equations for computing the update later
        self.relaxed_alg = vertcat(self.relaxed_alg, self.model.alg[self.relax_algebraic_index])

        # include the penalization term in the objective
        self.problem.L += (mtimes(nu_alg.T, self.model.alg[self.relax_algebraic_index])
                           + self.mu_sym / 2. * mtimes(self.model.alg[self.relax_algebraic_index].T,
                                                       self.model.alg[self.relax_algebraic_index]))

        # include the relaxed y_sym as controls
        self.problem.include_control(self.model.y_sym[self.relax_algebraic_var_index],
                                     u_max=self.problem.y_max[self.relax_algebraic_var_index],
                                     u_min=self.problem.y_min[self.relax_algebraic_var_index])
        self.problem.remove_algebraic(self.model.y_sym[self.relax_algebraic_var_index],
                                      self.model.alg[self.relax_algebraic_index])

    def _relax_states_constraints(self):
        for i in range(self.model.n_x):
            if self.problem.x_max[i] != inf or self.problem.x_min[i] != -inf:
                y_x = SX.sym('y_x_' + str(i))
                nu_y_x = SX.sym('nu_y_x_' + str(i))

                self.nu_sym = vertcat(self.nu_sym, nu_y_x)

                new_alg = y_x - self.model.x_sym[i]
                self.problem.L += dot(nu_y_x.T, new_alg) + self.mu_sym / 2.0 * dot(new_alg.T, new_alg)

                self.relaxed_alg = vertcat(self.relaxed_alg, new_alg)
                self.problem.include_control(y_x, u_min=self.problem.x_min[i], u_max=self.problem.x_max[i])
                self.problem.x_max[i] = inf
                self.problem.x_min[i] = -inf
                self.n_relax += 1

    def _relax_inequalities(self):
        raise Exception('Not implemented')

    # endregion
    # ==============================================================================

    # ==============================================================================
    # region NU
    # ==============================================================================
    def _parametrize_nu(self, nu_sym=None, n_r=None):
        if nu_sym is None:
            nu_sym = self.nu_sym
        if n_r is None:
            n_r = self.n_relax

        nu_pol, nu_par = self.create_variable_polynomial_approximation(n_r, self.degree, 'nu')

        self.nu_pol = vertcat(self.nu_pol, nu_pol)
        self.nu_par = vertcat(self.nu_par, nu_par)

        self.problem.replace_variable(nu_sym, nu_pol)
        self.problem.model.include_theta(vec(nu_par))

        return nu_pol, nu_par

    def create_nu_initial_guess(self, n_r=None):
        if n_r is None:
            n_r = self.n_relax
        nu = create_constant_theta(constant=0, dimension=n_r * self.degree, finite_elements=self.finite_elements)
        return nu

    def _create_nu_update_func(self):
        v = self.opt_problem.x
        x_var, y_var, u_var, eta, p_opt, theta_opt = self.ocp_solver.discretizer.unpack_decision_variables(v)
        par = MX.sym('par', self.model.n_p)
        theta = dict([(i, vec(MX.sym('theta_' + repr(i), self.model.n_theta)))
                      for i in range(self.finite_elements)])
        theta_var = vertcat(*[theta[i] for i in range(self.finite_elements)])

        time_dict = defaultdict(dict)

        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['f_nu'] = self.time_interpolation_nu[el]
            time_dict[el]['f_relax_alg'] = self.time_interpolation_nu[el]

        functions = defaultdict(dict)
        for el in range(self.finite_elements):
            func_rel_alg = self.model.convert_expr_from_tau_to_time(
                substitute(self.relaxed_alg, self.model.u, self.model.u_expr),
                self.time_breakpoints[el],
                self.time_breakpoints[el + 1])
            nu_time_dependent = self.model.convert_expr_from_tau_to_time(self.nu_pol,
                                                                         self.time_breakpoints[el],
                                                                         self.time_breakpoints[el + 1])

            f_nu = Function('f_nu', self.model.all_sym, [nu_time_dependent])
            f_relax_alg = Function('f_relax_alg', self.model.all_sym, [func_rel_alg])

            functions['f_nu'][el] = f_nu
            functions['f_relax_alg'][el] = f_relax_alg

        results = self.ocp_solver.discretizer.get_system_at_given_times(x_var, y_var, u_var, time_dict, p=par,
                                                                        theta=theta,
                                                                        functions=functions)

        mu_mx = par[find_variables_indices_in_vector(self.mu_sym, self.model.p)]
        new_nu = []
        rel_alg = []
        for el in range(self.finite_elements):
            new_nu_k = []
            rel_alg_k = []
            for j in range(self.degree):
                nu_kj = results[el]['f_nu'][j]
                rel_alg_kj = results[el]['f_relax_alg'][j]
                new_nu_k = horzcat(new_nu_k, nu_kj + mu_mx * rel_alg_kj)
                rel_alg_k = horzcat(rel_alg_k, rel_alg_kj)
            new_nu.append(vec(new_nu_k))
            rel_alg.append(vec(rel_alg_k))

        output = new_nu + rel_alg
        self.new_nu_func = Function('nu_update_function', [v, par, theta_var], output)

    def _compute_new_nu_and_error(self, p=None, theta=None, raw_solution_dict=None):
        if raw_solution_dict is None:
            raw_solution_dict = {}
        if theta is None:
            theta = {}
        if p is None:
            p = []

        if self.new_nu_func is None:
            self._create_nu_update_func()

        if not self._debug_skip_update_nu:
            raw_decision_variables = raw_solution_dict['x']
            theta_vector = vertcat(*[theta[i] for i in range(self.finite_elements)])
            output = self.new_nu_func(raw_decision_variables, p, theta_vector)
            new_nu = output[:self.finite_elements]
            rel_alg = output[self.finite_elements:2 * self.finite_elements]

            self.nu_tilde = dict([(i, new_nu[i]) for i in range(self.finite_elements)])
            self.alg_violation = dict([(i, rel_alg[i]) for i in range(self.finite_elements)])

            error = max([mmax(fabs(rel_alg[i])) for i in range(self.finite_elements)])

            print("Violation: ", error)
        else:
            error = 0
        return error

    @staticmethod
    def join_nu_to_theta(theta, nu):
        if theta is not None:
            return join_thetas(theta, nu)
        else:
            return nu

    def _update_mu(self):
        if not self._debug_skip_update_mu:
            self.mu = min(self.mu_max, self.mu * self.beta)

    # endregion
    # ==============================================================================

    # ==============================================================================
    # SOLVE
    # ==============================================================================

    def call_solver(self, initial_guess=None, p=None, theta=None, x_0=None, last_u=None, initial_guess_dict=None):
        if self.opt_problem is None:
            self.ocp_solver.create_optimization_problem()
            self.opt_problem = self.ocp_solver.opt_problem

        if x_0 is None:
            x_0 = self.problem.x_0

        if not vertcat(x_0).numel() == self.model.n_x:
            raise Exception('Size of given x_0 (or obtained from problem.x_0) is different from model.n_x, '
                            'x_0.numel() = {}, model.n_x = {}'.format(vertcat(x_0).numel(), self.model.n_x))

        # parameters
        if p is None:
            if self.problem.n_p_opt == self.model.n_p:
                p = repmat(0, self.problem.n_p_opt)
            elif self.problem.model.n_p - 1 > 0:
                raise Exception("A parameter 'p' of size {} should be given.".format(self.problem.model.n_p))
            else:
                p = []

        # theta
        if theta is None:
            if self.problem.n_theta_opt == self.model.n_theta:
                theta = create_constant_theta(0, self.problem.n_theta_opt, self.finite_elements)
            elif self.problem.model.n_theta - self.n_relax * self.degree > 0:
                raise Exception("A parameter 'theta' of size {} should be given".format(self.problem.model.n_theta))

        # last control
        if not self.last_control_as_parameter and last_u is not None:
            raise warnings.warn('solution_method.last_control_as_parameter is False, but last_u was passed. last_u will'
                                ' be ignored.')
        else:
            if last_u is not None:
                if isinstance(last_u, list):
                    last_u = vertcat(*last_u)
            elif self.problem.last_u is not None:
                last_u = self.problem.last_u

        it = 0

        t1 = time.time()
        while True:
            if self.nu_tilde is not None:
                self.nu = self.nu_tilde

            theta_k = self.join_nu_to_theta(theta, self.nu)
            p_k = vertcat(p, self.mu)

            # build the optimization parameter vector
            if theta_k is not None:
                theta_k_vector = vertcat(*[theta_k[i] for i in range(self.finite_elements)])
                par = vertcat(p_k, theta_k_vector)
            else:
                par = p_k

            # initial condition
            if self.initial_condition_as_parameter:
                par = vertcat(par, x_0)

            if last_u is not None:
                par = vertcat(par, last_u)
            elif self.problem.last_u is not None:
                par = vertcat(par, self.problem.last_u)

            # if initial_guess_dict is None:
            #     if initial_guess is None:
            #         if self.initial_guess_heuristic == 'simulation':
            #             initial_guess = self.ocp_solver.discretizer.create_initial_guess_with_simulation(p=p_k,
            #                                                                                              theta=theta_k)
            #         elif self.initial_guess_heuristic == 'problem_info':
            #             initial_guess = self.ocp_solver.discretizer.create_initial_guess(p_k, theta_k)
            #         else:
            #             raise ValueError(
            #                 'initial_guess_heuristic did not recognized, available options: "simulation" and '
            #                 '"problem_info". Given: {}'.format(self.initial_guess_heuristic))

            if initial_guess_dict is None:
                args = dict(initial_guess=initial_guess, p=par)
            else:
                args = dict(initial_guess=initial_guess_dict['x'], p=par,
                            lam_x=initial_guess_dict['lam_x'], lam_g=initial_guess_dict['lam_g'])

            # solve the optimization problem
            raw_solution_dict = self.opt_problem.solve(**args)
            initial_guess_dict = raw_solution_dict

            it += 1

            error = self._compute_new_nu_and_error(p=p_k, theta=theta_k, raw_solution_dict=raw_solution_dict)
            self._update_mu()
            self.last_violation_error = error

            if it == self.max_iter:
                break
            else:
                pass
        print('Solution time: {}'.format(time.time() - t1))
        print('Approximation error: {}'.format(error))
        return raw_solution_dict, p, theta, x_0, last_u

    def create_optimization_result(self, raw_solution_dict, p=None, theta=None, x_0=None):
        if p is None:
            p = []
        if theta is None:
            theta = {}
        p = vertcat(p, self.mu)
        theta = self.join_nu_to_theta(self.nu, theta)

        optimization_result = super(AugmentedLagrangian, self).create_optimization_result(raw_solution_dict, p=p,
                                                                                          theta=theta, x_0=x_0)

        optimization_result.other_data['nu']['values'] = [[self.unvec(self.nu[el])[:, d] for d in range(self.degree)]
                                                          for el in range(self.finite_elements)]
        optimization_result.other_data['nu']['time'] = self.time_interpolation_nu

        optimization_result.other_data['alg_violation']['values'] = [[self.unvec(self.alg_violation[el])[:, d]
                                                                      for d in range(self.degree)]
                                                                     for el in range(self.finite_elements)]
        optimization_result.other_data['alg_violation']['time'] = self.time_interpolation_nu

        optimization_result.violation_error = self.last_violation_error
        return optimization_result
