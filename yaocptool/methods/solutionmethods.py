# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 12:27:29 2016

@author: marco
"""

from yaocptool import config

# if not 'casadi' in sys.modules:
from casadi import SX, MX, DM, inf, repmat, vertcat, collocation_points, \
    substitute, pi, diag, integrator, vec, nlpsol, \
    Function, linspace, horzcat, dot, gradient, jacobian, mtimes, \
    reshape
import matplotlib.pyplot as plt
import types

from yaocptool.methods.multipleshooting import MultipleShootingScheme
from yaocptool.methods.collocationscheme import CollocationScheme
from yaocptool.modelling_classes.model_classes import SystemModel
# from CONFIG import SOLVER_OPTIONS

solver = "nlpsol"

# _obj_scaling_factor = 1e-3
obj_scaling_factor = 1


# integrator_options= {
#        'abstol' : 1e-12, # abs. tolerance
#        'reltol' :  1e-12 # rel. tolerance
#    }

class SolutionMethodsBase:
    def __init__(self, **kwargs):
        self.solver = None
        # self.model  #type: SystemModel
        self.integrator_type = 'implicit'
        self.solution_method = 'multiple_shooting'
        self.degree = 4
        self.degree_control = 1
        self.finite_elements = 10
        self.prepared = False
        self.discretization_method = 'multiple-shooting'
        # self.discretization_method = 'collocation'

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        self.defineDiscretizationMethods()

    @property
    def model(self):
        return self.problem.model

    def defineDiscretizationMethods(self):
        if self.discretization_method == 'multiple-shooting':
            methods_dict = MultipleShootingScheme().getMethods()
            for method_name in methods_dict:
                method = methods_dict[method_name]
                setattr(self, method_name, types.MethodType(method, self))
        if self.discretization_method == 'collocation':
            methods_dict = CollocationScheme().getMethods()
            for method_name in methods_dict:
                method = methods_dict[method_name]
                setattr(self, method_name, types.MethodType(method, self))

    def createLagrangianPolynomialBasis(self, degree, starting_index=0, tau=None):
        cp = "radau"  # Radau collocation points
        if tau == None:
            tau = self.model.tau_sym  # Collocation point
        tau_root = [0] + collocation_points(degree, cp)  # All collocation time points

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        L_list = []
        for j in range(starting_index, degree + 1):
            L = 1
            for j2 in range(starting_index, degree + 1):
                if j2 != j:
                    L *= (tau - tau_root[j2]) / (tau_root[j] - tau_root[j2])
            L_list.append(L)

        return tau, L_list

    def createVariablePolynomialApproximation(self, size, degree, name='var_appr', tau=None, point_at_t0=False):
        if tau == None:
            tau = self.model.tau_sym  # Collocation point

        if degree == 1:
            points = SX.sym(name, size, degree)
            par = vec(points)
            u_pol = points
        else:
            if point_at_t0:
                points = SX.sym(name, size, degree + 1)
                tau, ell_list = self.createLagrangianPolynomialBasis(degree, 0)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree + 1)])
            else:
                points = SX.sym(name, size, degree)
                tau, ell_list = self.createLagrangianPolynomialBasis(degree, 1)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree)])
            par = vec(points)

        return u_pol, par

    def createControlApproximation(self):
        degree = self.degree_control
        if self.parametrized_control == False:
            if type(degree) == dict:
                raise Exception('Not implemented')
            else:
                u_pol, self.model.u_par = self.createVariablePolynomialApproximation(self.model.Nu, degree, 'u_ij')
            self.u_pol = u_pol
            self.model.control_function = u_pol

        else:
            u_pol = self.u_pol
        return u_pol

    def createCostState(self):
        if not self.hasCostState:
            self.problem.createCostState()

            self.hasCostState = True

    def includeAdjointStates(self):
        Nx_old = self.model.Nx
        Ny_old = self.model.Nyz

        lamb = SX.sym('lamb', Nx_old)
        nu = SX.sym('nu', Ny_old)

        self.problem.eta = SX.sym('eta', self.problem.N_h_final)

        self.problem.H = self.problem.L + dot(lamb, self.model.ode) + dot(nu, self.model.all_alg)

        ldot = -gradient(self.problem.H, self.model.x_sym)
        alg_eq = gradient(self.problem.H, self.model.yz_sym)

        self.problem.includeState(lamb, ldot, suppress=True)
        self.model.hasAdjointVariables = True

        self.problem.includeAlgebraic(nu, alg_eq)

        self.problem.h_final = vertcat(self.problem.h_final,
                                       self.model.lamb_sym - gradient(self.problem.V, self.model.x_sys_sym)
                                       - mtimes(jacobian(self.problem.h_final, self.model.x_sys_sym).T,
                                                self.problem.eta))

    def joinXandU(self, X, U):
        V = []
        for k in range(self.finite_elements + 1):
            V.append(X[k])
            if k != self.finite_elements:
                V.append(U[k])
        return vertcat(*V)

    def unvec(self, vect, degree=None):
        if degree == None:
            degree = self.degree
        n_lines = vect.numel() / self.degree
        return reshape(vect, n_lines, self.degree)

    def convertFromTimeToTau(self, dae_sys, t_k, t_kp1):
        raise Exception
        t = self.model.t_sym
        tau = self.model.tau_sym

        h = t_kp1 - t_k
        dae_sys['ode'] = substitute(dae_sys['ode'], tau, (t - t_k) / h)
        if 'alg' in dae_sys:
            dae_sys['alg'] = substitute(dae_sys['alg'], tau, (t - t_k) / h)

    def convertFromTauToTime(self, dae_sys, t_k, t_kp1):
        raise Exception
        t = self.model.t_sym
        tau = self.model.tau_sym

        h = t_kp1 - t_k
        dae_sys['ode'] = substitute(dae_sys['ode'], tau, (t - t_k) / h)
        if 'alg' in dae_sys:
            dae_sys['alg'] = substitute(dae_sys['alg'], tau, (t - t_k) / h)

    def createIntegrator(self, dae_sys, options):
        raise Exception
        if self.integrator_type == 'implicit':
            if self.model.system_type == 'ode':
                I = integrator("I", "cvodes", dae_sys, options)
            else:
                I = integrator("I", "idas", dae_sys, options)
        else:
            if self.model.system_type == 'ode':
                I = self.explicitIntegrator('explicitIntegrator', 'rk4', dae_sys, options)
            else:
                raise Exception('explicit integrator not implemented')
        return I

    def joinThetas(self, *args):
        new_theta = {}
        all_keys = []
        for theta in args:
            all_keys.extend(theta.keys())
        all_keys = set(all_keys)

        for i in all_keys:
            new_theta[i] = []
            for theta in args:
                if i in theta:
                    theta1_value = theta[i]
                else:
                    theta1_value = []

                new_theta[i] = vertcat(new_theta[i], theta1_value)

        return new_theta

    def createConstantTheta(self, constant=0, dimension=1, degree=None, finite_elements=None):
        if finite_elements == None:
            finite_elements = self.finite_elements
        if degree == None:
            degree = self.degree

        theta = {}
        for i in xrange(finite_elements):
            theta[i] = vec(constant * DM.ones(dimension, degree))

        return theta

        # ==============================================================================

    # SOLVE
    # ==============================================================================

    def getSolver(self, initial_condition_as_parameter=False):
        ''' 
            all_mx = [p, theta, x_0]
        '''

        if not self.prepared:
            self.prepare()
            self.prepared = True

        if self.solver == None:
            self.initial_condition_as_parameter = initial_condition_as_parameter
            if self.model.Np + self.model.Ntheta > 0 or self.initial_condition_as_parameter:
                p_mx = MX.sym('p', self.model.Np)

                theta_mx = MX.sym('theta_', self.model.Ntheta, self.finite_elements)
                theta = dict([(i, vec(theta_mx[:, i])) for i in range(self.finite_elements)])

                all_mx = vertcat(p_mx, vec(theta_mx))
                if initial_condition_as_parameter:
                    p_mx_x_0 = MX.sym('x_0_p', self.model.Nx)
                    all_mx = vertcat(all_mx, p_mx_x_0)
                else:
                    p_mx_x_0 = None

                nlp_prob, nlp_call = self.discretize(p=p_mx, x_0=p_mx_x_0, theta=theta)

                nlp_prob['p'] = all_mx

            else:
                nlp_prob, nlp_call = self.discretize()

            self.nlp_prob = nlp_prob
            self.nlp_call = nlp_call
            self.solver = self.createNumSolver(nlp_prob)

        return self.callSolver

    def callSolver(self, initial_guess=None, p=[], theta=None, x_0=[]):
        if initial_guess == None:
            initial_guess = self.createInitialGuess()

        if theta != None:
            par = vertcat(p, *theta.values())
        else:
            par = p
        if self.initial_condition_as_parameter:
            par = vertcat(par, x_0)
        sol = self.solver(x0=initial_guess, p=par, lbg=self.nlp_call['lbg'], ubg=self.nlp_call['ubg'],
                          lbx=self.nlp_call['lbx'], ubx=self.nlp_call['ubx'])

        return sol['x']

    def createNumSolver(self, nlp_prob):
        solver = nlpsol('solver', 'ipopt', nlp_prob, config.SOLVER_OPTIONS['nlpsol_options'])
        return solver

    def solveNumProblem(self, nlp_prob, nlp_call, initial_guess=0):
        solver = nlpsol('solver', 'ipopt', nlp_prob, config.SOLVER_OPTIONS['nlpsol_options'])

        #        solver_call ={'x0':x0, 'lbg':0, 'ubg':0, 'lbx':num_prob['lbx'], 'ubx':num_prob['ubx']}
        nlp_call['x0'] = initial_guess

        V_sol = solver(**nlp_call)['x']
        return V_sol

    def solve_raw(self, initial_guess=None, p=[]):
        if not self.prepared:
            self.prepare()
            self.prepared = True

        solver = self.getSolver()

        V_sol = solver(initial_guess=initial_guess)
        return V_sol

    def solve(self, initial_guess=None, p=[]):
        V_sol = self.solve_raw(initial_guess, p)

        X, U = self.splitXandU(V_sol)
        return X, U, V_sol

    # ==============================================================================
    # PLOT AND SIMULAT
    # ==============================================================================

    def plot(self, X, Y, U, plot_list, t_states=None):
        if t_states == None:
            t_states = linspace(self.problem.t_0, self.problem.t_f, self.finite_elements + 1)

        if isinstance(plot_list, int):
            plot_list = [plot_list]
        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_states, horzcat(*X)[i, :].T)
                plt.legend(['x[' + `i` + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_states[:len(U)], horzcat(*Y)[i, :].T)
                plt.legend(['y[' + `i` + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.plot(t_states[:len(U)], horzcat(*U)[i, :].T)
                plt.legend(['u[' + `i` + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k +=1
        # plt.ion()
        plt.show()

    def simulate(self, X, U, sub_elements=5, t_0=None, t_f=None, p=[], theta={}, integrator_type='implicit',
                 time_division='linear'):
        finite_elements = len(X) - 1
        if theta == {}:
            theta = dict([(i, []) for i in xrange(finite_elements)])

        if t_0 == None:
            t_0 = self.problem.t_0
        if t_f == None:
            t_f = self.problem.t_f

        t_list = [float(t) for t in (linspace(t_0, t_f, finite_elements + 1)).full()]
        micro_t = [t_0]

        #        F_y = Function('f_y', [self.model.x_sym, self.model.yz_sym,  self.model.t_sym,
        #                                    self.model.p_sym, self.model.theta_sym,
        #                                    self.model.u_par], [self.model.alg])
        # Simualtion
        micro_X = [X[0]]
        micro_Y = []
        micro_U = []
        x_0 = X[0]
        for k in xrange(finite_elements):
            #            x_0 = X[k]
            dae_sys = self.model.getDAESystem()
            self.model.convertFromTauToTime(dae_sys, t_list[k], t_list[k + 1])
            func_u = self.model.convertExprFromTauToTime(self.model.control_function, t_list[k], t_list[k + 1])

            F_u = Function('f_u', [self.model.x_sym, self.model.yz_sym, self.model.t_sym,
                                   self.model.p_sym, self.model.theta_sym,
                                   self.model.u_par], [func_u])

            if time_division == 'linear':
                micro_t_k = list(linspace(t_list[k], t_list[k + 1], sub_elements + 1).full())
            else:
                tau_list = [0] + collocation_points(sub_elements)
                dt = t_list[k + 1] - t_list[k]
                mapping = lambda tau: (t_list[k] + tau * dt)
                micro_t_k = map(mapping, tau_list)
            micro_t += micro_t_k[1:]
            par = vertcat(p, theta[k], U[k])
            x_f, y_f = self.model.simulateInterval(x_0, t_list[k], t_list[k + 1], micro_t_k[1:], p=par, dae_sys=dae_sys,
                                                   integrator_type=integrator_type)
            micro_X.extend(x_f)
            micro_Y.extend(y_f)
            #            x_f.insert(0,x_0)
            for j in range(sub_elements):
                micro_U.append(F_u(x_f[j], y_f[j], float(micro_t_k[j + 1]), p, theta[k], U[k]))

            x_0 = x_f[-1]

        return micro_X, micro_Y, micro_U, micro_t

    def plotSimulate(self, X, U, plot_list, sub_elements=5, p=[], theta={}, integrator_type=None,
                     time_division='linear'):
        if integrator_type == None:
            integrator_type = self.integrator_type
        micro_X, micro_Y, micro_U, micro_t = self.simulate(X, U, sub_elements=sub_elements,
                                                           t_0=None, t_f=None, p=p, theta=theta,
                                                           integrator_type=integrator_type,
                                                           time_division=time_division)
        self.plot(micro_X, micro_Y, micro_U, plot_list, t_states=micro_t)
        return micro_X, micro_Y, micro_U, micro_t

    def stepForward(self):
        pass


