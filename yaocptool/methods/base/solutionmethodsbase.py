import matplotlib.pyplot as plt
from casadi import SX, MX, DM, vertcat, collocation_points, \
    vec, nlpsol, \
    Function, linspace, horzcat, dot, gradient, jacobian, mtimes, \
    reshape
from typing import List, Tuple
import copy

from yaocptool.methods.classic.multipleshooting import MultipleShootingScheme
from yaocptool import config
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.classic.collocationscheme import CollocationScheme
from yaocptool.modelling.ocp import OptimalControlProblem


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
        self._problem = problem
        self.problem = copy.copy(self._problem)  # type: OptimalControlProblem
        self.integrator_type = 'implicit'
        self.solution_method = 'multiple_shooting'
        self.solution_class = ''
        self.degree = 4
        self.degree_control = 1
        self.finite_elements = 10
        self.prepared = False
        self.discretization_scheme = 'multiple-shooting'
        # self.discretization_scheme = 'collocation'
        self.discretizer = None  # type: DiscretizationSchemeBase
        self.initial_condition_as_parameter = False
        self.parametrized_control = False
        self.nlp_prob = {}
        self.nlp_call = {}
        self.nlpsol_opts = {}

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        for k in config.SOLVER_OPTIONS['nlpsol_options']:
            if k not in self.nlpsol_opts:
                self.nlpsol_opts[k] = config.SOLVER_OPTIONS['nlpsol_options'][k]

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
    def split_x_and_u(self):
        return self.discretizer.split_x_and_u

    @property
    def split_x_y_and_u(self):
        return self.discretizer.split_x_y_and_u

    @staticmethod
    def collocation_points(degree, cp='radau', with_zero=False):
        # type: (int, str, bool) -> List[int]
        if with_zero:
            return [0] + collocation_points(degree, cp)  # All collocation time points
        else:
            return collocation_points(degree, cp)  # All collocation time points

    def create_lagrangian_polynomial_basis(self, degree, starting_index=0, tau=None):
        if tau is None:
            tau = self.model.tau_sym  # symbolic variable

        tau_root = self.collocation_points(degree, with_zero=True)  # All collocation time points

        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        l_list = []
        for j in range(starting_index, degree + 1):
            ell = 1
            for j2 in range(starting_index, degree + 1):
                if j2 != j:
                    ell *= (tau - tau_root[j2]) / (tau_root[j] - tau_root[j2])
            l_list.append(ell)

        return tau, l_list

    def create_variable_polynomial_approximation(self, size, degree, name='var_appr', tau=None, point_at_t0=False):
        # type: (int, int, str, SX, bool) -> Tuple[DM, DM]
        if tau is None:
            tau = self.model.tau_sym  # Collocation point

        if degree == 1:
            points = SX.sym(name, size, degree)
            par = vec(points)
            u_pol = points
        else:
            if point_at_t0:
                points = SX.sym(name, size, degree + 1)
                tau, ell_list = self.create_lagrangian_polynomial_basis(degree, 0, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree + 1)])
            else:
                points = SX.sym(name, size, degree)
                tau, ell_list = self.create_lagrangian_polynomial_basis(degree, 1, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree)])
            par = vec(points)

        return u_pol, par

    def create_control_approximation(self):
        degree = self.degree_control
        if not self.parametrized_control:
            if type(degree) == dict:
                raise Exception('Not implemented')
            else:
                u_pol, self.model.u_par = self.create_variable_polynomial_approximation(self.model.n_u, degree, 'u_ij')
            self.model.u_func = u_pol
        else:
            u_pol = self.model.u_func

        return u_pol

    def _create_cost_state(self):
        if not self.hasCostState:
            self.problem.create_cost_state()
            self.hasCostState = True

    def include_adjoint_states(self):

        lamb = SX.sym('lamb', self.model.n_x)
        nu = SX.sym('nu', self.model.n_yz)

        self.problem.eta = SX.sym('eta', self.problem.n_h_final)

        self.problem.H = self.problem.L + dot(lamb, self.model.ode) + dot(nu, self.model.all_alg)

        ldot = -gradient(self.problem.H, self.model.x_sym)
        alg_eq = gradient(self.problem.H, self.model.yz_sym)

        self.problem.include_state(lamb, ldot, suppress=True)
        self.model.hasAdjointVariables = True

        self.problem.include_algebraic(nu, alg_eq)

        self.problem.h_final = vertcat(self.problem.h_final,
                                       self.model.lamb_sym - gradient(self.problem.V, self.model.x_sys_sym)
                                       - mtimes(jacobian(self.problem.h_final, self.model.x_sys_sym).T,
                                                self.problem.eta))

    def join_x_and_u(self, x, u):
        v = []
        for k in range(self.finite_elements + 1):
            v.append(x[k])
            if k != self.finite_elements:
                v.append(u[k])
        return vertcat(*v)

    def unvec(self, vector, degree=None):
        """
        Unvectorize 'vector' a vectorized matrix, assuming that it was a matrix with 'degree' number of columns
        :type vector: DM a vecotr (flattened matrix)
        :type degree: int
        """
        if degree is None:
            degree = self.degree
        n_lines = vector.numel() / degree
        return reshape(vector, n_lines, degree)

    @staticmethod
    def join_thetas(*args):
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

    def create_constant_theta(self, constant=0, dimension=1, degree=None, finite_elements=None):
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

    def prepare(self):
        self.problem.pre_solve_check()

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
        if self.model.n_p + self.model.n_theta > 0 or self.initial_condition_as_parameter:
            p_mx = MX.sym('p', self.model.n_p)

            theta_mx = MX.sym('theta_', self.model.n_theta, self.finite_elements)
            theta = dict([(i, vec(theta_mx[:, i])) for i in range(self.finite_elements)])

            all_mx = vertcat(p_mx, vec(theta_mx))
            if initial_condition_as_parameter:
                p_mx_x_0 = MX.sym('x_0_p', self.model.n_x)
                all_mx = vertcat(all_mx, p_mx_x_0)
            else:
                p_mx_x_0 = None

            nlp_prob, nlp_call = self.discretizer.discretize(p=p_mx, x_0=p_mx_x_0, theta=theta)

            nlp_prob['p'] = all_mx
        else:
            nlp_prob, nlp_call = self.discretizer.discretize()

        self.nlp_prob = nlp_prob
        self.nlp_call = nlp_call

        solver = nlpsol('solver', 'ipopt', nlp_prob, self.nlpsol_opts)
        return solver

    def call_solver(self, initial_guess=None, p=None, theta=None, x_0=None):
        if x_0 is None:
            x_0 = []
        if p is None:
            p = []
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

    def solve_raw(self, initial_guess=None, p=None, theta=None, x_0=None):
        if p is None:
            p = []
        if theta is None:
            theta = {}
        if x_0 is None:
            x_0 = []
        if not self.prepared:
            self.prepare()
            self.prepared = True

        solution_dict = self.get_solver()(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return solution_dict

    def solve(self, initial_guess=None, p=None, theta=None, x_0=None):
        # type: (object, list, dict, list) -> OptimizationResult
        if p is None:
            p = []
        if theta is None:
            theta = {}
        if x_0 is None:
            x_0 = []
        raw_solution_dict = self.solve_raw(initial_guess=initial_guess, p=p, theta=theta, x_0=x_0)
        return self.create_optimization_result(raw_solution_dict, p, theta, x_0=x_0)

    def create_optimization_result(self, raw_solution_dict, p=None, theta=None, x_0=None):
        if x_0 is None:
            x_0 = self.problem.x_0
        if theta is None:
            theta = {}
        if p is None:
            p = []

        optimization_result = OptimizationResult()

        # From the solution_method
        for attr in ['finite_elements', 'degree', 'degree_control', 'time_breakpoints', 'discretization_scheme']:
            attr_value = getattr(self, attr)
            setattr(optimization_result, attr, attr_value)
        optimization_result.method_name = self.__class__.__name__

        # Initial condition, theta, and parameters used
        optimization_result.x_0 = x_0
        optimization_result.theta = theta
        optimization_result.p = p

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

    def plot(self, x, y, u, plot_list, t_states=None):
        if t_states is None:
            t_states = linspace(self.problem.t_0, self.problem.t_f, self.finite_elements + 1)

        if isinstance(plot_list, int):
            plot_list = [plot_list]
        for k, entry in enumerate(plot_list):
            fig = plt.figure(k)
            if 'x' in entry:
                for i in entry['x']:
                    plt.plot(t_states, horzcat(*x)[i, :].T)
                plt.legend(['x[' + repr(i) + ']' for i in entry['x']])
            if 'y' in entry:
                for i in entry['y']:
                    plt.plot(t_states[:len(u)], horzcat(*y)[i, :].T)
                plt.legend(['y[' + repr(i) + ']' for i in entry['y']])

            if 'u' in entry:
                for i in entry['u']:
                    plt.plot(t_states[:len(u)], horzcat(*u)[i, :].T)
                plt.legend(['u[' + repr(i) + ']' for i in entry['u']])
            plt.grid()
            axes = fig.axes
            axes[0].ticklabel_format(useOffset=False)
            k += 1
        # plt.ion()
        plt.show()

    def simulate(self, x, u, sub_elements=5, t_0=None, t_f=None, p=None, theta=None, integrator_type='implicit',
                 time_division='linear'):
        if theta is None:
            theta = {}
        if p is None:
            p = []
        finite_elements = len(x) - 1
        if theta == {}:
            theta = dict([(i, []) for i in range(finite_elements)])

        if t_0 is None:
            t_0 = self.problem.t_0
        if t_f is None:
            t_f = self.problem.t_f

        t_list = [float(t) for t in (linspace(t_0, t_f, finite_elements + 1)).full()]
        micro_t = [t_0]

        # Simualtion
        micro_x = [x[0]]
        micro_y = []
        micro_u = []
        x_0 = x[0]
        for k in range(finite_elements):
            dae_sys = self.model.get_dae_system()
            self.model.convert_dae_sys_from_tau_to_time(dae_sys, t_list[k], t_list[k + 1])
            func_u = self.model.convert_expr_from_tau_to_time(self.model.u_func, t_list[k], t_list[k + 1])

            f_u = Function('f_u', [self.model.x_sym, self.model.yz_sym, self.model.t_sym,
                                   self.model.p_sym, self.model.theta_sym,
                                   self.model.u_par], [func_u])

            if time_division == 'linear':
                micro_t_k = list(linspace(t_list[k], t_list[k + 1], sub_elements + 1).full())
            else:
                tau_list = self.collocation_points(sub_elements, with_zero=True)
                micro_t_k = [self.time_breakpoints[k] + self.delta_t*tau_list[j] for j in range(self.degree)]
            micro_t += micro_t_k[1:]
            par = vertcat(p, theta[k], u[k])
            x_f, y_f = self.model.simulate_interval(x_0, t_list[k], micro_t_k[1:], p=par, dae_sys=dae_sys,
                                                    integrator_type=integrator_type)
            micro_x.extend(x_f)
            micro_y.extend(y_f)
            #            x_f.insert(0,x_0)
            for j in range(sub_elements):
                micro_u.append(f_u(x_f[j], y_f[j], float(micro_t_k[j + 1]), p, theta[k], u[k]))

            x_0 = x_f[-1]

        return micro_x, micro_y, micro_u, micro_t

    def plot_simulate(self, x, u, plot_list, sub_elements=5, p=None, theta=None, integrator_type=None,
                      time_division='linear'):
        if p is None:
            p = []
        if theta is None:
            theta = {}
        if integrator_type is None:
            integrator_type = self.integrator_type
        micro_x, micro_y, micro_u, micro_t = self.simulate(x, u, sub_elements=sub_elements,
                                                           t_0=None, t_f=None, p=p, theta=theta,
                                                           integrator_type=integrator_type,
                                                           time_division=time_division)
        self.plot(micro_x, micro_y, micro_u, plot_list, t_states=micro_t)
        return micro_x, micro_y, micro_u, micro_t

    def step_forward(self):
        pass
