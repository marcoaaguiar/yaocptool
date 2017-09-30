import matplotlib.pyplot as plt
from casadi import SX, MX, DM, vertcat, collocation_points, \
    vec, nlpsol, \
    Function, linspace, horzcat, dot, gradient, jacobian, mtimes, \
    reshape
from typing import List
from yaocptool.methods.classic.multipleshooting import MultipleShootingScheme

from yaocptool import config
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.classic.collocationscheme import CollocationScheme
from yaocptool.modelling_classes.ocp import OptimalControlProblem


# TODO: fix PEP 8

class SolutionMethodsBase(object):
    def __init__(self, problem, **kwargs):
        """
        :param problem: OptimalControlProblem
        :param integrator_type: str
        :param solution_method: str
        :param degree: int
        :param discretization_scheme: str
        """
        self.solver = None
        self.problem = problem  # type: OptimalControlProblem
        self.integrator_type = 'implicit'
        self.solution_method = 'multiple_shooting'
        self.degree = 4
        self.degree_control = 1
        self.finite_elements = 10
        self.prepared = False
        self.discretization_scheme = 'multiple-shooting'
        # self.discretization_scheme = 'collocation'
        self.discretizer = None  # type: DiscretizationSchemeBase
        self.initial_condition_as_parameter = False
        self.nlp_prob = {}
        self.nlp_call = {}

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.discretization_scheme == 'multiple-shooting':
            self.discretizer = MultipleShootingScheme(self)
        elif self.discretization_scheme == 'collocation':
            self.discretizer = CollocationScheme(self)

    @property
    def model(self):
        return self.problem.model

    @property
    def delta_t(self):
        return float(self.problem.t_f - self.problem.t_0) / self.finite_elements

    @property
    def time_breakpoints(self):
        return [self.delta_t * k for k in range(self.finite_elements + 1)]

    @property
    def splitXandU(self):
        return self.discretizer.splitXandU

    @property
    def splitXYandU(self):
        return self.discretizer.splitXYandU

    @staticmethod
    def collocation_points(degree, cp='radau', with_zero=False):
        # type: (int, str, bool) -> List[int]
        if with_zero:
            return [0] + collocation_points(degree, cp)  # All collocation time points
        else:
            return collocation_points(degree, cp)  # All collocation time points

    def createLagrangianPolynomialBasis(self, degree, starting_index=0, tau=None):
        if tau is None:
            tau = self.model.tau_sym  # symbolic variable

        tau_root = self.collocation_points(degree, with_zero=True)  # All collocation time points

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        l_list = []
        for j in range(starting_index, degree + 1):
            L = 1
            for j2 in range(starting_index, degree + 1):
                if j2 != j:
                    L *= (tau - tau_root[j2]) / (tau_root[j] - tau_root[j2])
            l_list.append(L)

        return tau, l_list

    def createVariablePolynomialApproximation(self, size, degree, name='var_appr', tau=None, point_at_t0=False):
        if tau is None:
            tau = self.model.tau_sym  # Collocation point

        if degree == 1:
            points = SX.sym(name, size, degree)
            par = vec(points)
            u_pol = points
        else:
            if point_at_t0:
                points = SX.sym(name, size, degree + 1)
                tau, ell_list = self.createLagrangianPolynomialBasis(degree, 0, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree + 1)])
            else:
                points = SX.sym(name, size, degree)
                tau, ell_list = self.createLagrangianPolynomialBasis(degree, 1, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree)])
            par = vec(points)

        return u_pol, par

    def createControlApproximation(self):
        degree = self.degree_control
        if not self.parametrized_control:
            if type(degree) == dict:
                raise Exception('Not implemented')
            else:
                u_pol, self.model.u_par = self.createVariablePolynomialApproximation(self.model.Nu, degree, 'u_ij')
            self.u_pol = u_pol
            self.model.control_function = u_pol

        else:
            u_pol = self.u_pol
        return u_pol

    def _create_cost_state(self):
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
        if degree is None:
            degree = self.degree
        n_lines = vect.numel() / degree
        return reshape(vect, n_lines, degree)

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
        if finite_elements is None:
            finite_elements = self.finite_elements
        if degree is None:
            degree = self.degree

        theta = {}
        for i in range(finite_elements):
            theta[i] = vec(constant * DM.ones(dimension, degree))

        return theta

    # ==============================================================================
    # SOLVE
    # ==============================================================================

    def get_solver(self, initial_condition_as_parameter=False):
        """
            all_mx = [p, theta, x_0]
        """

        if not self.prepared:
            self.prepare()
            self.prepared = True

        if self.solver is None:
            self.solver = self.create_solver(initial_condition_as_parameter)

        return self.call_solver

    def create_solver(self, initial_condition_as_parameter):
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

            nlp_prob, nlp_call = self.discretizer.discretize(p=p_mx, x_0=p_mx_x_0, theta=theta)

            nlp_prob['p'] = all_mx
        else:
            nlp_prob, nlp_call = self.discretizer.discretize()

        self.nlp_prob = nlp_prob
        self.nlp_call = nlp_call
        solver = nlpsol('solver', 'ipopt', nlp_prob, config.SOLVER_OPTIONS['nlpsol_options'])
        return solver

    def call_solver(self, initial_guess=None, p=[], theta=None, x_0=[]):
        if initial_guess is None:
            initial_guess = self.discretizer.create_initial_guess()

        if theta is not None:
            par = vertcat(p, *theta.values())
        else:
            par = p
        if self.initial_condition_as_parameter:
            par = vertcat(par, x_0)
        sol = self.solver(x0=initial_guess, p=par, lbg=self.nlp_call['lbg'], ubg=self.nlp_call['ubg'],
                          lbx=self.nlp_call['lbx'], ubx=self.nlp_call['ubx'])

        return sol

    def callSolver(self, initial_guess=None, p=[], theta=None, x_0=[]):
        raise Exception('method changed to call_solver')

    def solve_raw(self, initial_guess=None, p=[], theta={}, x_0=[]):
        if not self.prepared:
            self.prepare()
            self.prepared = True

        solution_dict = self.get_solver()(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return solution_dict

    def solve(self, initial_guess=None, p=[], theta={}, x_0=[]):
        # type: (object, list, dict, list) -> OptimizationResult
        raw_solution_dict = self.solve_raw(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return self.create_optimization_result(raw_solution_dict)

    def create_optimization_result(self, raw_solution_dict):
        optimization_result = OptimizationResult()
        # From the solution_method
        for attr in ['finite_elements', 'degree', 'degree_control', 'time_breakpoints', 'discretization_scheme']:
            attr_value = getattr(self, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.method_name = self.__class__.__name__

        # From the problem
        for attr in ['t_0', 't_f']:
            attr_value = getattr(self.problem, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.problem_name = self.problem.name

        self.discretizer.set_data_to_optimization_result_from_raw_data(optimization_result, raw_solution_dict)

        return optimization_result

    # ==============================================================================
    # PLOT AND SIMULAT
    # ==============================================================================

    def plot(self, X, Y, U, plot_list, t_states=None):
        if t_states is None:
            t_states = linspace(self.problem.t_0, self.problem.t_f, self.finite_elements + 1)

        if isinstance(plot_list, int):
            plot_list = [plot_list]
        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_states, horzcat(*X)[i, :].T)
                plt.legend(['x[' + repr(i) + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_states[:len(U)], horzcat(*Y)[i, :].T)
                plt.legend(['y[' + repr(i) + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.plot(t_states[:len(U)], horzcat(*U)[i, :].T)
                plt.legend(['u[' + repr(i) + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        # plt.ion()
        plt.show()

    def simulate(self, X, U, sub_elements=5, t_0=None, t_f=None, p=[], theta={}, integrator_type='implicit',
                 time_division='linear'):
        finite_elements = len(X) - 1
        if theta == {}:
            theta = dict([(i, []) for i in range(finite_elements)])

        if t_0 is None:
            t_0 = self.problem.t_0
        if t_f is None:
            t_f = self.problem.t_f

        t_list = [float(t) for t in (linspace(t_0, t_f, finite_elements + 1)).full()]
        micro_t = [t_0]

        # Simualtion
        micro_X = [X[0]]
        micro_Y = []
        micro_U = []
        x_0 = X[0]
        for k in range(finite_elements):
            dae_sys = self.model.getDAESystem()
            self.model.convertFromTauToTime(dae_sys, t_list[k], t_list[k + 1])
            func_u = self.model.convertExprFromTauToTime(self.model.control_function, t_list[k], t_list[k + 1])

            F_u = Function('f_u', [self.model.x_sym, self.model.yz_sym, self.model.t_sym,
                                   self.model.p_sym, self.model.theta_sym,
                                   self.model.u_par], [func_u])

            if time_division == 'linear':
                micro_t_k = list(linspace(t_list[k], t_list[k + 1], sub_elements + 1).full())
            else:
                tau_list = self.collocation_points(sub_elements, with_zero=True)
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
        if integrator_type is None:
            integrator_type = self.integrator_type
        micro_X, micro_Y, micro_U, micro_t = self.simulate(X, U, sub_elements=sub_elements,
                                                           t_0=None, t_f=None, p=p, theta=theta,
                                                           integrator_type=integrator_type,
                                                           time_division=time_division)
        self.plot(micro_X, micro_Y, micro_U, plot_list, t_states=micro_t)
        return micro_X, micro_Y, micro_U, micro_t

    def stepForward(self):
        pass
