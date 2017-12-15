from warnings import warn

from casadi import SX, MX, DM, vertcat, collocation_points, \
    vec, nlpsol, \
    dot, gradient, jacobian, mtimes, \
    reshape
from typing import List, Tuple

from yaocptool import config, create_constant_theta, join_thetas
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.classic.collocationscheme import CollocationScheme
from yaocptool.methods.classic.multipleshooting import MultipleShootingScheme


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
        self.problem = problem
        # self.problem = copy.copy(self._problem)  # type: OptimalControlProblem
        self.integrator_type = 'implicit'
        self.solution_class = ''
        self.degree = 3
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
        else:
            raise Exception("Discretization scheme not recognized: '{}'. Available options: 'multiple-shooting'"
                            " and 'collocation'".format(self.discretization_scheme))

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
        warn('Use yaocptool.join_theta')
        return join_thetas(*args)

    def create_constant_theta(self, constant=0, dimension=1, finite_elements=None):
        if finite_elements is None:
            finite_elements = self.finite_elements

        return create_constant_theta(constant, dimension, finite_elements)

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

        # From model
        optimization_result.x_names = [self.model.x_sym[i].name() for i in range(self.model.n_x)]
        optimization_result.y_names = [self.model.y_sym[i].name() for i in range(self.model.n_y)]
        optimization_result.z_names = [self.model.z_sym[i].name() for i in range(self.model.n_z)]
        optimization_result.u_names = [self.model.u_sym[i].name() for i in range(self.model.n_u)]

        self.discretizer.set_data_to_optimization_result_from_raw_data(optimization_result, raw_solution_dict)

        return optimization_result

    def step_forward(self):
        pass
