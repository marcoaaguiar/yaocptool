# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import time
from collections import defaultdict

from casadi import SX, DM, inf, vertcat, dot, vec, Function, MX, horzcat

from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


# TODO: Fix PEP 8
# TODO: update_nu calculate error

class AugmentedLagrange(SolutionMethodsBase):
    """
        For a minmization problem in the form
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

        super(AugmentedLagrange, self).__init__(problem, **kwargs)

        self.mu = self.mu_0

        # RELAXATION
        if self.model.Ny > 0 and self.relax_algebraic:
            self._relax_algebraic_equations()

        if self.model.Nyz > 0 and self.relax_connecting_equations:
            self._relax_connecting_equations()

        if self.model.Nz > 0 and self.relax_external_algebraic:
            self._relax_external_algebraic_equations()

        if self.relax_state_bounds:
            self._relax_states_constraints()

        # CREATE THE OCP SOLVER
        self.model.includeParameter(self.mu_sym)

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
    def splitXYandU(self):
        return self.ocp_solver.splitXYandU

    @property
    def splitXandU(self):
        return self.ocp_solver.splitXandU

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
        self.problem.includeControl(self.model.y_sym, u_max=self.problem.y_max, u_min=self.problem.y_min)
        self.problem.removeAlgebraic(self.model.y_sym, self.model.alg)

    def _relax_external_algebraic_equations(self):
        nu_alg = SX.sym('AL_nu_alg_z', self.model.alg_z.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.alg_z) + self.mu_sym / 2. * dot(self.model.alg_z, self.model.alg_z)

        self.Nr += self.model.alg_z.size1()

        self._save_relaxed_equation(self.model.alg_z)
        z_without_con_z = self.model.removeVariablesFromVector(self.model.con_z, vertcat(self.model.z_sym))
        z_without_con_z_indices = self.model.find_variables_indices_in_vector(z_without_con_z, self.model.z_sym)

        self.problem.includeControl(z_without_con_z, u_max=self.problem.z_max[z_without_con_z_indices],
                                    u_min=self.problem.z_min[z_without_con_z_indices])
        self.problem.removeExternalAlgebraic(z_without_con_z, self.model.alg_z)

    def _relax_connecting_equations(self):
        nu_alg = SX.sym('AL_nu_alg_con', self.model.con.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.con) + self.mu_sym / 2. * dot(self.model.con, self.model.con)

        self.Nr += self.model.con.size1()

        self._save_relaxed_equation(self.model.con)
        con_z_ind = self.model.find_variables_indices_in_vector(self.model.con_z, self.model.z_sym)

        self.problem.includeControl(self.model.con_z, u_max=self.problem.z_max[con_z_ind],
                                    u_min=self.problem.z_min[con_z_ind])

        self.problem.removeConnectingEquations(var=self.model.con_z, eq=self.model.con)

    def _relax_states_constraints(self):
        for i in range(self.model.Nx):
            if self.problem.x_max[i] != inf or self.problem.x_min[i] != -inf:
                y_x = SX.sym('y_x_' + str(i))
                nu_y_x = SX.sym('nu_y_x_' + str(i))

                self.nu_sym = vertcat(self.nu_sym, nu_y_x)

                new_alg = y_x - self.model.x_sym[i]
                self.problem.L += dot(nu_y_x.T, new_alg) + self.mu_sym / 2.0 * dot(new_alg.T, new_alg)

                self._save_relaxed_equation(new_alg)
                self.problem.includeControl(y_x, u_min=self.problem.x_min[i], u_max=self.problem.x_max[i])
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

        nu_pol, nu_par = self.createVariablePolynomialApproximation(n_r, self.degree, 'nu')

        self.nu_pol = vertcat(self.nu_pol, nu_pol)
        self.nu_par = vertcat(self.nu_par, nu_par)

        self.problem.replace_variable(nu_sym, nu_pol, 'other')
        self.problem.model.includeTheta(vec(nu_par))

        return nu_pol, nu_par

    def create_nu_initial_guess(self, n_r=None):
        if n_r is None:
            n_r = self.Nr
        nu = self.createConstantTheta(constant=0, dimension=n_r, degree=self.degree,
                                      finite_elements=self.finite_elements)
        return nu

    def _create_nu_update_func(self):
        v, x_var, y_var, u_var, eta = self.ocp_solver.discretizer.create_nlp_symbolic_variables()
        par = MX.sym('par', self.model.Np)
        theta = dict([(i, vec(MX.sym('theta_' + repr(i), self.Nr, self.degree))) for i in range(self.finite_elements)])
        theta_var = vertcat(*theta.values())

        col_points = self.collocation_points(self.degree, with_zero=False)

        new_nu = []
        error = 0
        time_dict = defaultdict(dict)

        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['f_nu'] = [self.time_breakpoints[el] + self.delta_t * col_points[j] for j in
                                     range(self.degree)]
            time_dict[el]['f_relax_alg'] = [self.time_breakpoints[el] + self.delta_t * col_points[j] for j in
                                            range(self.degree)]

        functions = defaultdict(dict)
        for el in range(self.finite_elements):
            func_rel_alg = self.model.convertExprFromTauToTime(self.relaxed_alg, self.time_breakpoints[el],
                                                               self.time_breakpoints[el + 1])
            nu_time_dependent = self.model.convertExprFromTauToTime(self.nu_pol, self.time_breakpoints[el],
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
        self.new_nu_func = Function('new_nu_funct', [v, par, theta_var], output)

    def _update_nu(self, p, theta, raw_solution_dict):
        if not self._debug_skip_update_nu:
            if self.new_nu_func is None:
                self._create_nu_update_func()
            raw_decision_variables = raw_solution_dict['x']
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
            return self.joinThetas(theta, nu)
        else:
            return nu

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

    def stepForward(self):
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

    def solve_raw(self, initial_guess=None, p=None, theta=None, x_0=None):
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
            raw_solution_dict = solver(initial_guess, p=p_k, theta=theta_k, x_0=x_0)
            initial_guess = raw_solution_dict['x']
            x, u = self.splitXandU(initial_guess)
            it += 1

            if it == self.max_iter:
                self.last_solution = (x, u)
                break
            else:
                error = self._update_nu(p=self.mu, theta=self.nu, raw_solution_dict=raw_solution_dict)
                self._update_mu()

        print('Solution time: {}'.format(time.time() - t1))
        print('Approximation error: {}'.format(error))
        return raw_solution_dict

    def solve(self, initial_guess=None, p=None, theta=None, x_0=None):
        # type: (object, list, dict, list) -> OptimizationResult
        if x_0 is None:
            x_0 = []
        if theta is None:
            theta = {}
        if p is None:
            p = []
        raw_solution_dict = self.solve_raw(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return self.ocp_solver.create_optimization_result(raw_solution_dict)

    # ==============================================================================
    #     PLOT/SIMULATE
    # ==============================================================================

    def simulate(self, x, u, sub_elements=5, t_0=None, t_f=None, p=None, theta=None, integrator_type='implicit',
                 time_division='linear'):
        if theta is None:
            theta = {}
        if p is None:
            p = []
        if t_0 is None:
            t_0 = self.problem.t_0
        if t_f is None:
            t_f = self.problem.t_f

        par = vertcat(p, self.mu)
        nu_theta = self.join_nu_to_theta(theta, self.nu)

        micro_x, micro_y, micro_u, micro_t = self.ocp_solver.simulate(x, u, sub_elements, t_0, t_f, par, nu_theta,
                                                                      integrator_type=integrator_type)
        return micro_x, micro_y, micro_u, micro_t

    def _update_mu(self):
        if not self._debug_skip_update_mu:
            self.mu = min(self.mu_max, self.mu * self.beta)
