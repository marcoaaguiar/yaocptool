# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import time
from collections import defaultdict

from casadi import SX, DM, inf, vertcat, dot, vec, Function, MX, horzcat

from yaocptool import find_variables_indices_in_vector, remove_variables_from_vector
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


# TODO: Fix PEP 8
# TODO: update_nu calculate error

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
        if solver_options is None:
            solver_options = {}

        self.degree = 4
        self.degree_control = 4

        self.Nr = 0
        self.nu_sym = []
        self.mu_sym = SX.sym('mu')
        self.nu_par = vertcat([])
        self.nu_pol = vertcat([])

        self.max_iter = 3
        self.mu_0 = 10.
        self.beta = 4.
        self.mu_max = self.mu_0 * self.beta ** 3
        self.nu = None
        self.new_nu_func = None  # type: Function

        self.tol = 1e-4
        self.last_solution = ()

        self.solver = None
        self.ocp_solver = None  # type: SolutionMethodsBase
        self.solver_initialized = False

        self.relax_algebraic = True
        self.relax_external_algebraic = True
        self.relax_connecting_equations = False
        self.relax_state_bounds = False

        self._debug_skip_parametrize = False
        self._debug_skip_initialize = False
        self._debug_skip_update_nu = False
        self._debug_skip_update_mu = False

        super(AugmentedLagrangian, self).__init__(problem, **kwargs)

        self.mu = self.mu_0

        # RELAXATION
        if self.model.n_y > 0 and self.relax_algebraic:
            self._relax_algebraic_equations()

        if self.model.n_yz > 0 and self.relax_connecting_equations:
            self._relax_connecting_equations()

        if self.model.n_z > 0 and self.relax_external_algebraic:
            self._relax_external_algebraic_equations()

        if self.relax_state_bounds:
            self._relax_states_constraints()

        # CREATE THE OCP SOLVER
        self.model.include_parameter(self.mu_sym)

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

        if not self._debug_skip_initialize:
            if not self._debug_skip_parametrize:
                self._parametrize_nu()

            if self.nu is None:
                self.nu = self.create_nu_initial_guess()

            # Initialize OCP solver

            self.ocp_solver = ocp_solver_class(self.problem, **solver_options)  # type: SolutionMethodsBase

    # region # PROPERTY
    @property
    def model(self):
        return self.problem.model

    @property
    def relaxed_alg(self):
        return self.model.relaxed_alg

    @property
    def split_x_y_and_u(self):
        return self.ocp_solver.split_x_y_and_u

    @property
    def split_x_and_u(self):
        return self.ocp_solver.split_x_and_u

    @property
    def time_interpolation_nu(self):
        col_points = self.collocation_points(self.degree, with_zero=False)
        return [[self.time_breakpoints[el] + self.delta_t * col_points[j] for j in
                 range(self.degree)] for el in range(self.finite_elements)]

    # endregion

    # ==============================================================================
    # region RELAX

    def _save_relaxed_equation(self, alg):
        self.model.relaxed_alg = vertcat(self.relaxed_alg, alg)

    def _relax_algebraic_equations(self):
        nu_alg = SX.sym('AL_nu_alg', self.model.alg.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.alg) + self.mu_sym / 2. * dot(self.model.alg, self.model.alg)

        self.Nr += self.model.alg.size1()

        self._save_relaxed_equation(self.model.alg)
        self.problem.include_control(self.model.y_sym, u_max=self.problem.y_max, u_min=self.problem.y_min)
        self.problem.remove_algebraic(self.model.y_sym, self.model.alg)

    def _relax_external_algebraic_equations(self):
        nu_alg = SX.sym('AL_nu_alg_z', self.model.alg_z.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.alg_z) + self.mu_sym / 2. * dot(self.model.alg_z, self.model.alg_z)

        self.Nr += self.model.alg_z.size1()

        self._save_relaxed_equation(self.model.alg_z)
        z_without_con_z = remove_variables_from_vector(self.model.con_z, vertcat(self.model.z_sym))
        z_without_con_z_indices = find_variables_indices_in_vector(z_without_con_z, self.model.z_sym)

        self.problem.include_control(z_without_con_z, u_max=self.problem.z_max[z_without_con_z_indices],
                                     u_min=self.problem.z_min[z_without_con_z_indices])
        self.problem.remove_external_algebraic(z_without_con_z, self.model.alg_z)

    def _relax_connecting_equations(self):
        nu_alg = SX.sym('AL_nu_alg_con', self.model.con.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.con) + self.mu_sym / 2. * dot(self.model.con, self.model.con)

        self.Nr += self.model.con.size1()

        self._save_relaxed_equation(self.model.con)
        con_z_ind = find_variables_indices_in_vector(self.model.con_z, self.model.z_sym)

        self.problem.include_control(self.model.con_z, u_max=self.problem.z_max[con_z_ind],
                                     u_min=self.problem.z_min[con_z_ind])

        self.problem.remove_connecting_equations(var=self.model.con_z, eq=self.model.con)

    def _relax_states_constraints(self):
        for i in range(self.model.n_x):
            if self.problem.x_max[i] != inf or self.problem.x_min[i] != -inf:
                y_x = SX.sym('y_x_' + str(i))
                nu_y_x = SX.sym('nu_y_x_' + str(i))

                self.nu_sym = vertcat(self.nu_sym, nu_y_x)

                new_alg = y_x - self.model.x_sym[i]
                self.problem.L += dot(nu_y_x.T, new_alg) + self.mu_sym / 2.0 * dot(new_alg.T, new_alg)

                self._save_relaxed_equation(new_alg)
                self.problem.include_control(y_x, u_min=self.problem.x_min[i], u_max=self.problem.x_max[i])
                self.problem.x_max[i] = inf
                self.problem.x_min[i] = -inf
                self.Nr += 1

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
            n_r = self.Nr

        nu_pol, nu_par = self.create_variable_polynomial_approximation(n_r, self.degree, 'nu')

        self.nu_pol = vertcat(self.nu_pol, nu_pol)
        self.nu_par = vertcat(self.nu_par, nu_par)

        self.problem.replace_variable(nu_sym, nu_pol, 'other')
        self.problem.model.include_theta(vec(nu_par))

        return nu_pol, nu_par

    def create_nu_initial_guess(self, n_r=None):
        if n_r is None:
            n_r = self.Nr
        nu = self.create_constant_theta(constant=0, dimension=n_r * self.degree, finite_elements=self.finite_elements)
        return nu

    def _create_nu_update_func(self):
        v, x_var, y_var, u_var, eta = self.ocp_solver.discretizer.create_nlp_symbolic_variables()
        par = MX.sym('par', self.model.n_p)
        theta = dict([(i, vec(MX.sym('theta_' + repr(i), self.model.n_theta)))
                      for i in range(self.finite_elements)])
        theta_var = vertcat(*theta.values())

        new_nu = []
        error = 0
        time_dict = defaultdict(dict)

        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['f_nu'] = self.time_interpolation_nu[el]
            time_dict[el]['f_relax_alg'] = self.time_interpolation_nu[el]

        functions = defaultdict(dict)
        for el in range(self.finite_elements):
            func_rel_alg = self.model.convert_expr_from_tau_to_time(self.relaxed_alg, self.time_breakpoints[el],
                                                                    self.time_breakpoints[el + 1])
            nu_time_dependent = self.model.convert_expr_from_tau_to_time(self.nu_pol, self.time_breakpoints[el],
                                                                         self.time_breakpoints[el + 1])

            f_nu = Function('f_nu', self.model.all_sym, [nu_time_dependent])
            f_relax_alg = Function('f_relax_alg', self.model.all_sym, [func_rel_alg])

            functions['f_nu'][el] = f_nu
            functions['f_relax_alg'][el] = f_relax_alg

        results = self.ocp_solver.discretizer.get_system_at_given_times(x_var, y_var, u_var, time_dict, p=par,
                                                                        theta=theta,
                                                                        functions=functions)
        for el in range(self.finite_elements):
            new_nu_k = []
            for j in range(self.degree):
                nu_kj = results[el]['f_nu'][j]
                rel_alg_kj = results[el]['f_relax_alg'][j]
                new_nu_k = horzcat(new_nu_k, nu_kj + self.mu * rel_alg_kj)
            new_nu.append(vec(new_nu_k))

        output = new_nu + [error]
        self.new_nu_func = Function('nu_update_function', [v, par, theta_var], output)

    def _update_nu(self, p=None, theta=None, raw_solution_dict=None):
        if raw_solution_dict is None:
            raw_solution_dict = {}
        if theta is None:
            theta = {}
        if p is None:
            p = []
        if not self._debug_skip_update_nu:
            if self.new_nu_func is None:
                self._create_nu_update_func()
            raw_decision_variables = raw_solution_dict['x']
            # noinspection PyCallingNonCallable
            output = self.new_nu_func(raw_decision_variables, p, vertcat(*theta.values()))
            new_nu = output[:-1]
            error = output[-1]
            print("Violation: ", error)

            self.nu = dict([(i, new_nu[i]) for i in range(self.finite_elements)])
        else:
            error = 0
        return error

    def join_nu_to_theta(self, theta, nu):
        if theta is not None:
            return self.join_thetas(theta, nu)
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

    def get_solver(self, initial_condition_as_parameter=False):
        self.get_ocp_solver(initial_condition_as_parameter)
        return self.solve_raw

    def get_ocp_solver(self, initial_condition_as_parameter=False):
        self.solver = self.ocp_solver.get_solver(initial_condition_as_parameter=initial_condition_as_parameter)

    def step_forward(self):
        raise NotImplementedError
        # X, U = self.last_solution
        # error = self._update_nu(self.mu, self.nu, raw_solution_dict=raw_solution_dict)
        # self.mu = min(self.mu_max, self.mu * self.beta)
        # new_nu = {}
        # for i in range(1, self.finite_elements):
        #     new_nu[i - 1] = self.nu[i]
        # new_nu[self.finite_elements - 1] = self.nu[self.finite_elements - 1]
        # self.nu = new_nu
        # self.mu = self.mu_0

    def solve_raw(self, initial_guess=None, p=None, theta=None, x_0=None, last_u=None):
        if x_0 is None:
            x_0 = []
        if theta is None:
            theta = {}
        if p is None:
            p = []
        if self.solver_initialized:
            if len(DM(x_0).full()) > 0:
                initial_condition_as_parameter = True
            else:
                initial_condition_as_parameter = False
            self.get_ocp_solver(initial_condition_as_parameter)
            self.solver_initialized = True
        solver = self.ocp_solver.get_solver()

        it = 0
        error = -1

        t1 = time.time()
        while True:
            theta_k = self.join_nu_to_theta(theta, self.nu)
            p_k = vertcat(p, self.mu)
            raw_solution_dict = solver(initial_guess, p=p_k, theta=theta_k, x_0=x_0, last_u=last_u)
            initial_guess = raw_solution_dict['x']
            x, u = self.split_x_and_u(initial_guess)
            it += 1

            if it == self.max_iter:
                self.last_solution = (x, u)
                break
            else:
                error = self._update_nu(p=p_k, theta=theta_k, raw_solution_dict=raw_solution_dict)
                self._update_mu()

        print('Solution time: {}'.format(time.time() - t1))
        print('Approximation error: {}'.format(error))
        return raw_solution_dict

    def create_optimization_result(self, raw_solution_dict, p=None, theta=None, x_0=None):
        if p is None:
            p = []
        if theta is None:
            theta = {}
        p = vertcat(p, self.mu)
        theta = self.join_nu_to_theta(self.nu, theta)
        result = self.ocp_solver.create_optimization_result(raw_solution_dict, p, theta, x_0)
        result.other_data['nu']['values'] = [[self.unvec(self.nu[el])[:, d] for d in range(self.degree)] for el in
                                             range(self.finite_elements)]
        result.other_data['nu']['times'] = self.time_interpolation_nu
        return result
