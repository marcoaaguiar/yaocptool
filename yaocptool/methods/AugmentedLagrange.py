# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import time

from casadi import SX, DM, inf, vertcat, dot, collocation_points, vec, Function, linspace, MX, horzcat

from solutionmethodsbase import SolutionMethodsBase


class AugmentedLagrange(SolutionMethodsBase):
    '''
        For a minmization problem in the form
            min f(x,u) = \int L(x,u) dt
            s.t.: \dot{x} = f(x,u),
            g_ineq (x,u) \leq 0
        
        Transforms the problem in a sequence of solution of the problem
            min f(x,u) = \int L(x,u) -\mu \sum \log(-g_ineq(x,u)) dt
            s.t.: \dot{x} = f(x,u),
    '''

    def __init__(self, problem, Ocp_solver_class, solver_options={}, **kwargs):
        # SolutionMethodsBase.__init__(self, **kwargs)
        self.problem = problem

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
        self.new_nu_funct = None
        self.tol = 1e-4
        self.m = 0

        self.solver = None
        self.ocp_solver = None  # :type solution_method: yaocptool.methods.solutionmethodsbase.SolutionMethodsBase
        self.solver_initialized = False

        self.relax_algebraic = True
        self.relax_external_algebraic = True
        self.relax_connecting_equations = False
        self.relax_state_bounds = False

        self.parametrize = True
        self.initialize = True

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        self.mu = self.mu_0

        # RELAXATION
        if self.model.Ny > 0 and self.relax_algebraic:
            self.relax_algebraic_equations()

        if self.model.Nyz > 0 and self.relax_connecting_equations:
            self.relax_connecting_equations()

        if self.model.Nz > 0 and self.relax_external_algebraic:
            self.relax_external_algebraic_equations()

        if self.relax_state_bounds:
            self.relax_states_constraints()

        # CREATE THE OCP SOLVER
        self.model.includeParameter(self.mu_sym)

        for attr in ['degree', 'finite_elements', 'degree_control']:
            if attr in solver_options and attr in kwargs and solver_options[attr] != kwargs[attr]:
                exc_mess = "Trying to pass attribute '{}' for '{}' and '{}' that are not equal: {} != {}"
                raise Exception(exc_mess.format(attr, self.__class__.__name__, Ocp_solver_class.__name__, kwargs[attr],
                                                solver_options[attr]))
            elif attr in solver_options:
                setattr(self, attr, solver_options[attr])
            else:
                solver_options[attr] = getattr(self, attr)

        solver_options['integrator_type'] = self.integrator_type

        if self.initialize:
            if self.parametrize:
                self.parametrize_nu()

            if self.nu is None:
                self.initialize_nu_values()

            # Initialize OCP solver

            self.ocp_solver = Ocp_solver_class(self.problem, **solver_options)

    @property
    def model(self):
        return self.problem.model

    @property
    def relaxed_alg(self):
        return self.model.relaxed_alg

    @property
    def joinXandU(self):
        return self.ocp_solver.joinXandU

    @property
    def splitXandU(self):
        return self.ocp_solver.splitXandU

    # ==============================================================================
    # RELAX
    # ==============================================================================

    def save_relaxed_equation(self, alg):
        self.model.relaxed_alg = vertcat(self.relaxed_alg, alg)

    def relax_algebraic_equations(self):
        nu_alg = SX.sym('AL_nu_alg', self.model.alg.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.alg) + self.mu_sym / 2. * dot(self.model.alg, self.model.alg)

        self.Nr += self.model.alg.size1()

        self.save_relaxed_equation(self.model.alg)
        self.problem.includeControl(self.model.y_sym, u_max=self.problem.y_max, u_min=self.problem.y_min)
        self.problem.removeAlgebraic(self.model.y_sym, self.model.alg)

    def relax_external_algebraic_equations(self):
        nu_alg = SX.sym('AL_nu_alg_z', self.model.alg_z.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.alg_z) + self.mu_sym / 2. * dot(self.model.alg_z, self.model.alg_z)

        self.Nr += self.model.alg_z.size1()

        self.save_relaxed_equation(self.model.alg_z)
        z_without_con_z = self.model.removeVariablesFromVector(self.model.con_z, vertcat(self.model.z_sym))
        z_without_con_z_indeces = self.model.findVariablesIndecesInVector(z_without_con_z, self.model.z_sym)

        self.problem.includeControl(z_without_con_z, u_max=self.problem.z_max[z_without_con_z_indeces],
                                    u_min=self.problem.z_min[z_without_con_z_indeces])
        self.problem.removeExternalAlgebraic(z_without_con_z, self.model.alg_z)

    def relax_connecting_equations(self):
        nu_alg = SX.sym('AL_nu_alg_con', self.model.con.size1())
        self.nu_sym = vertcat(self.nu_sym, nu_alg)
        self.problem.L += dot(nu_alg, self.model.con) + self.mu_sym / 2. * dot(self.model.con, self.model.con)

        self.Nr += self.model.con.size1()

        self.save_relaxed_equation(self.model.con)
        con_z_ind = self.model.findVariablesIndecesInVector(self.model.con_z, self.model.z_sym)

        self.problem.includeControl(self.model.con_z, u_max=self.problem.z_max[con_z_ind],
                                    u_min=self.problem.z_min[con_z_ind])

        self.problem.removeConnectingEquations(var=self.model.con_z, eq=self.model.con)

    def relax_states_constraints(self):
        for i in range(self.model.Nx):
            if self.problem.x_max[i] != inf or self.problem.x_min[i] != -inf:
                y_x = SX.sym('y_x_' + str(i))
                nu_y_x = SX.sym('nu_y_x_' + str(i))

                self.nu_sym = vertcat(self.nu_sym, nu_y_x)

                new_alg = y_x - self.model.x_sym[i]
                self.problem.L += dot(nu_y_x.T, new_alg) + self.mu_sym / 2.0 * dot(new_alg.T, new_alg)

                self.save_relaxed_equation(new_alg)
                self.problem.includeControl(y_x, u_min=self.problem.x_min[i], u_max=self.problem.x_max[i])
                self.problem.x_max[i] = inf
                self.problem.x_min[i] = -inf
                self.Nr += 1

    def relax_inequalities(self):
        raise Exception('Not implemented')

    # ==============================================================================
    # NU
    # ==============================================================================
    def parametrize_nu(self, nu_sym=None, n_r=None):
        if nu_sym is None:
            nu_sym = self.nu_sym
        if n_r is None:
            n_r = self.Nr

        nu_pol, nu_par = self.createVariablePolynomialApproximation(n_r, self.degree, 'nu')

        self.nu_pol = vertcat(self.nu_pol, nu_pol)
        self.nu_par = vertcat(self.nu_par, nu_par)

        self.problem.replaceVariable(nu_sym, nu_pol, 'other')
        self.problem.model.includeTheta(vec(nu_par))

        return nu_pol, nu_par

    def initialize_nu_values(self):
        nu = self.create_nu_initial_guess()
        self.nu = nu

    def create_nu_initial_guess(self, Nr=None):
        if Nr == None:
            Nr = self.Nr
        nu = self.createConstantTheta(constant=0, dimension=Nr, degree=self.degree,
                                      finite_elements=self.finite_elements)
        return nu

    def create_new_nu_generator(self):
        X_var = MX.sym('X', self.model.Nx, (self.finite_elements + 1))
        U_var = MX.sym('U', self.model.Nu * self.ocp_solver.degree_control, (self.finite_elements))
        par = MX.sym('par')

        x_error = SX.sym('x_error')
        theta = dict([(i, vec(MX.sym('theta_' + `i`, self.Nr, self.degree))) for i in xrange(self.finite_elements)])
        theta_var = vertcat(*theta.values())

        X = [X_var[:, i] for i in range((self.finite_elements + 1))]
        U = [U_var[:, i] for i in range(self.finite_elements)]

        t_list = self.time_breakpoints
        col_points = self.collocation_points(self.degree, with_zero=False)

        delta_t_list = [col_points[j] * self.delta_t for j in range(self.degree)]
        new_nu = []
        error = 0
        for k in range(self.finite_elements):
            x_0 = vertcat(X[k], DM.zeros(1))  # the last value is the initial state of the 'error state'
            dae_sys = self.model.getDAESystem()
            dae_sys['x'] = vertcat(dae_sys['x'], x_error)
            dae_sys['ode'] = vertcat(dae_sys['ode'], dot(self.relaxed_alg, self.relaxed_alg))

            self.model.convertFromTauToTime(dae_sys, t_list[k], t_list[k + 1])
            nu_time_dependent = self.model.convertExprFromTauToTime(self.nu_pol, t_list[k], t_list[k + 1])

            func_rel_alg = self.model.convertExprFromTauToTime(self.relaxed_alg, t_list[k], t_list[k + 1])
            F_nu = Function('F_nu', [self.model.x_sym, self.model.yz_sym, self.model.t_sym,
                                     self.model.p_sym, self.model.theta_sym,
                                     self.model.u_par], [nu_time_dependent])
            F_relax_alg = Function('F_rel_alg', [
                self.model.x_sym, self.model.yz_sym, self.model.t_sym,
                self.model.p_sym, self.model.theta_sym,
                self.model.u_par], [func_rel_alg])
            new_nu_k = []
            for j in range(self.degree):
                I = self.model.createIntegrator(dae_sys, {'t0': float(t_list[k]),
                                                          'tf': float(t_list[k] + delta_t_list[j])},
                                                integrator_type=self.integrator_type)

                sim = I(x0=x_0, p=vertcat(par, theta[k], U[k]))
                x_f = sim['xf']
                yz_f = sim['zf']
                nu_kj = F_nu(x_f[:-1], yz_f[:-1], float(t_list[k] + delta_t_list[j]),
                             par, theta[k], U[k])
                rel_alg_kj = F_relax_alg(x_f[:-1], yz_f[:-1], float(t_list[k] + delta_t_list[j]),
                                         par, theta[k], U[k])
                new_nu_k = horzcat(new_nu_k, nu_kj + self.mu * rel_alg_kj)

            error += x_f[-1]
            new_nu.append(vec(new_nu_k))

        output = new_nu + [error]
        self.new_nu_funct = Function('new_nu_funct', [X_var, U_var, par, theta_var], output)

    def calculateNewNu(self, X, U, p, theta):
        if self.new_nu_funct == None:
            self.create_new_nu_generator()
        output = self.new_nu_funct(horzcat(*X), horzcat(*U), p, vertcat(*theta.values()))
        new_nu = output[:-1]
        error = output[-1]
        print "Violation ", error

        nu = dict([(i, new_nu[i]) for i in range(self.finite_elements)])
        return nu, error

    def joinNuToTheta(self, theta, nu):
        if theta is not None:
            return self.joinThetas(theta, nu)
        else:
            return nu

    # ==============================================================================
    # SOLVE
    # ==============================================================================

    def get_solver(self, initial_condition_as_parameter=False):
        self.getOCPSolver(initial_condition_as_parameter)
        return self.solve_raw

    def getOCPSolver(self, initial_condition_as_parameter=False):
        self.solver = self.ocp_solver.get_solver(initial_condition_as_parameter=initial_condition_as_parameter)

    def stepForward(self):
        X, U = self.last_solution
        self.nu, error = self.calculateNewNu(X, U, self.mu, self.nu)
        self.mu = min(self.mu_max, self.mu * self.beta)
        new_nu = {}
        for i in range(1, self.finite_elements):
            new_nu[i - 1] = self.nu[i]
        new_nu[self.finite_elements - 1] = self.nu[self.finite_elements - 1]
        self.nu = new_nu

    #        self.mu = self.mu_0

    def solve_raw(self, initial_guess=None, p=[], theta={}, x_0=[]):
        if self.solver_initialized:
            if len(DM(x_0).full()) > 0:
                initial_condition_as_parameter = True
            else:
                initial_condition_as_parameter = False
            self.getOCPSolver(initial_condition_as_parameter)
            self.solver_initialized = True
        solver = self.ocp_solver.get_solver()

        it = 0

        t1 = time.time()
        while True:
            theta_k = self.joinNuToTheta(theta, self.nu)
            raw_solution_dict = solver(initial_guess, p=vertcat(p, self.mu), theta=theta_k, x_0=x_0)
            initial_guess = raw_solution_dict['x']
            X, U = self.splitXandU(initial_guess)
            it += 1

            if it == self.max_iter:
                self.last_solution = (X, U)
                break
            else:
                self.nu, error = self.calculateNewNu(X, U, self.mu, self.nu)
                self.mu = min(self.mu_max, self.mu * self.beta)

        print 'Solution time: ', time.time() - t1
        return raw_solution_dict

    def solve(self, initial_guess=None, p=[], theta={}, x_0=[]):
        raw_solution_dict = self.solve_raw(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return self.ocp_solver.create_optimization_result(raw_solution_dict)

        # ==============================================================================

    #     PLOT/SIMULATE
    # ==============================================================================

    #    def plotSimulate(self, X, U, plot_list, n_of_sub_elements =  5, p = [], theta = None, integrator_type = None):
    #    def plotSimulate(self, X, U, plot_list, n_of_sub_elements =  5, p = [], theta = None, integrator_type = None):
    #    def plotSimulate(self, X, U, plot_list, n_of_sub_elements =  5, p = [], theta = None, integrator_type = None):
    #        if integrator_type == None:
    #            integrator_type = self.integrator_type
    #        return self.ocp_solver.plotSimulate(X, U, plot_list, n_of_sub_elements, p = self.mu, theta = self.nu, integrator_type = integrator_type)

    def simulate(self, X, U, sub_elements=5, t_0=None, t_f=None, p=[], theta={}, integrator_type='implicit',
                 time_division='linear'):
        if t_0 is None:
            t_0 = self.problem.t_0
        if t_f is None:
            t_f = self.problem.t_f

        par = vertcat(p, self.mu)
        nu_theta = self.joinNuToTheta(theta, self.nu)

        micro_X, micro_Y, micro_U, micro_t = self.ocp_solver.simulate(X, U, sub_elements, t_0, t_f, par, nu_theta,
                                                                      integrator_type=integrator_type)
        return micro_X, micro_Y, micro_U, micro_t
