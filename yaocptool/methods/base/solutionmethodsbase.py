from casadi import SX, MX, vertcat, collocation_points, vec, reshape, repmat

from yaocptool import config, create_constant_theta
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.classic.collocationscheme import CollocationScheme
from yaocptool.methods.classic.multipleshooting import MultipleShootingScheme
from yaocptool.modelling import OptimalControlProblem


class SolutionMethodsBase(object):
    def __init__(self, problem, **kwargs):
        """
        :param OptimalControlProblem problem:
        :param str integrator_type: str
        :param str solution_method: str
        :param int degree: discretization polynomial degree
        :param int degree_control:
        :param str discretization_scheme: ('multiple-shooting' | 'collocation')
        :param str initial_guess_heuristic: 'simulation' or 'problem_info'
        """
        self.opt_problem = None
        self.problem = problem
        self.solution_class = ''
        self.prepared = False
        self.discretizer = None  # type: DiscretizationSchemeBase

        # Options
        self.degree = 3
        self.degree_control = 1
        self.finite_elements = 10
        self.integrator_type = 'implicit'
        self.discretization_scheme = 'multiple-shooting'
        self.initial_condition_as_parameter = True
        self.nlpsol_opts = {}
        self.initial_guess_heuristic = 'simulation'  # 'problem_info'

        # Internal variables
        self.parametrized_control = False

        self.nlp_prob = {}
        self.nlp_call = {}

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        for k in config.SOLVER_OPTIONS['nlpsol_options']:
            if k not in self.nlpsol_opts:
                self.nlpsol_opts[k] = config.SOLVER_OPTIONS['nlpsol_options'][k]

        if self.discretization_scheme == 'multiple-shooting':
            self.degree = 1
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
        if with_zero:
            return [0] + collocation_points(degree, cp)  # All collocation time points
        else:
            return collocation_points(degree, cp)  # All collocation time points

    def _create_lagrangian_polynomial_basis(self, degree, starting_index=0, tau=None):
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
        if not isinstance(name, list):
            name = [name + '_' + str(i) for i in range(size)]

        if tau is None:
            tau = self.model.tau_sym  # Collocation point

        if degree == 1:
            if size > 0:
                points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
            else:
                points = SX.sym('empty_sx', size, degree)
            par = vec(points)
            u_pol = points
        else:
            if point_at_t0:
                if size > 0:
                    points = vertcat(*[SX.sym(name[s], 1, degree + 1) for s in range(size)])
                else:
                    points = SX.sym('empty_sx', size, degree)
                tau, ell_list = self._create_lagrangian_polynomial_basis(degree, starting_index=0, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree + 1)])
            else:
                if size > 0:
                    points = vertcat(*[SX.sym(name[s], 1, degree) for s in range(size)])
                else:
                    points = SX.sym('empty_sx', size, degree)
                tau, ell_list = self._create_lagrangian_polynomial_basis(degree, starting_index=1, tau=tau)
                u_pol = sum([ell_list[j] * points[:, j] for j in range(0, degree)])
            par = vec(points)

        return u_pol, par

    def create_control_approximation(self):
        """Parametrize the control variable, accordingly to the 'degree_control' attribute.
        If degree_control == 1, then a piecewise constant control will be used (most common).
        If degree_control > 1, then a piecewise polynomial approximation will be used with order 'degree_control'.

        :return:
        """
        degree = self.degree_control
        if not self.parametrized_control:
            if type(degree) == dict:
                raise Exception('Not implemented')
            else:
                u_pol, self.model.u_par = self.create_variable_polynomial_approximation(self.model.n_u, degree,
                                                                                        name=self.model.u_names)
            self.model.u_func = u_pol
        else:
            u_pol = self.model.u_func

        return u_pol

    def unvec(self, vector, degree=None):
        """
        Unvectorize 'vector' a vectorized matrix, assuming that it was a matrix with 'degree' number of columns
        :type vector: DM a vector (flattened matrix)
        :type degree: int
        """
        if degree is None:
            degree = self.degree
        n_lines = vector.numel() / degree
        return reshape(vector, n_lines, degree)

    # ==============================================================================
    # SOLVE
    # ==============================================================================

    def prepare(self):
        self.problem.pre_solve_check()

    def create_optimization_problem(self):
        if not self.prepared:
            self.prepare()
            self.prepared = True

        has_parameters = (self.model.n_p + self.model.n_theta > 0 or self.initial_condition_as_parameter
                          or self.problem.last_u is not None)
        args = {}
        all_mx = []
        if has_parameters:
            p_mx = MX.sym('p', self.model.n_p)

            theta_mx = MX.sym('theta_', self.model.n_theta, self.finite_elements)
            theta = dict([(i, vec(theta_mx[:, i])) for i in range(self.finite_elements)])

            all_mx = vertcat(p_mx, vec(theta_mx))
            if self.initial_condition_as_parameter:
                p_mx_x_0 = MX.sym('x_0_p', self.model.n_x)
                all_mx = vertcat(all_mx, p_mx_x_0)
            else:
                p_mx_x_0 = None

            if self.problem.last_u is not None:
                p_last_u = MX.sym('last_u', self.model.n_u)
                all_mx = vertcat(all_mx, p_last_u)
            else:
                p_last_u = None

            args = dict(p=p_mx, x_0=p_mx_x_0, theta=theta, last_u=p_last_u)

        # Discretize the problem
        opt_problem = self.discretizer.discretize(**args)

        if has_parameters:
            opt_problem.include_parameter(all_mx)

        self.opt_problem = opt_problem

    def call_solver(self, initial_guess=None, p=None, theta=None, x_0=None, last_u=None, initial_guess_dict=None):
        if self.opt_problem is None:
            self.create_optimization_problem()

        if x_0 is None:
            x_0 = self.problem.x_0

        if not vertcat(x_0).numel() == self.model.n_x:
            raise Exception('Size of given x_0 (or obtained from problem.x_0) is different from model.n_x, '
                            'x_0.numel() = {}, model.n_x = {}'.format(vertcat(x_0).numel(), self.model.n_x))
        if p is None:
            if self.problem.n_p_opt == self.model.n_p:
                p = repmat(0, self.problem.n_p_opt)
            elif self.problem.model.n_p > 0:
                raise Exception("A parameter 'p' of size {} should be given".format(self.problem.model.n_p))

        if theta is None:
            if self.problem.n_theta_opt == self.model.n_theta:
                theta = create_constant_theta(0, self.problem.n_theta_opt, self.finite_elements)
            elif self.problem.model.n_theta > 0:
                raise Exception("A parameter 'theta' of size {} should be given".format(self.problem.model.n_theta))

        if theta is not None:
            par = vertcat(p, *theta.values())
        else:
            par = p
        if self.initial_condition_as_parameter:
            par = vertcat(par, x_0)
        if last_u is not None:
            if isinstance(last_u, list):
                last_u = vertcat(*last_u)
            par = vertcat(par, last_u)
        elif self.problem.last_u is not None:
            par = vertcat(par, self.problem.last_u)

        if initial_guess_dict is None:
            if initial_guess is None:
                if self.initial_guess_heuristic == 'simulation':
                    initial_guess = self.discretizer.create_initial_guess_with_simulation(p=p, theta=theta)
                elif self.initial_guess_heuristic == 'problem_info':
                    initial_guess = self.discretizer.create_initial_guess(p, theta)
                else:
                    raise ValueError('initial_guess_heuristic did not recognized, available options: "simulation" and '
                                     '"problem_info". Given: {}'.format(self.initial_guess_heuristic))
            args = dict(initial_guess=initial_guess, p=par)
        else:
            args = dict(initial_guess=initial_guess_dict['x'], p=par,
                        lam_x=initial_guess_dict['lam_x'], lam_g=initial_guess_dict['lam_g'])

        sol = self.opt_problem.solve(**args)
        return sol

    # def solve_raw(self, initial_guess=None, p=None, theta=None, x_0=None, last_u=None, initial_guess_dict=None):
    #     if isinstance(x_0, list):
    #         x_0 = vertcat(x_0)
    #
    #     solution_dict = self.call_solver(initial_guess=initial_guess, p=p, theta=theta,
    #                                      x_0=x_0, last_u=last_u,
    #                                      initial_guess_dict=initial_guess_dict)
    #     return solution_dict

    def solve(self, initial_guess=None, p=None, theta=None, x_0=None, last_u=None, initial_guess_dict=None):
        """

        :param initial_guess: Initial guess
        :param p: Parameters values
        :param theta: Theta values
        :param x_0: Initial condition value
        :param last_u: Last control value
        :param initial_guess_dict: Initial guess as dict
        :return: OptimizationResult
        """
        if isinstance(x_0, list):
            x_0 = vertcat(x_0)

        raw_solution_dict = self.call_solver(initial_guess=initial_guess, p=p, theta=theta,
                                             x_0=x_0, last_u=last_u,
                                             initial_guess_dict=initial_guess_dict)

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
        optimization_result.u_names = [self.model.u_sym[i].name() for i in range(self.model.n_u)]
        optimization_result.theta_opt_names = [self.problem.theta_opt[i].name()
                                               for i in range(self.problem.n_theta_opt)]

        self.discretizer.set_data_to_optimization_result_from_raw_data(optimization_result, raw_solution_dict)

        return optimization_result
