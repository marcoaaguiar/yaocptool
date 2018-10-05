from math import factorial, ceil
from itertools import product

import numpy as np
from casadi import DM, SX, Function, mtimes, chol, solve, vertcat, substitute, repmat, depends_on, inf, is_equal, \
    sqrt, fmax, diagcat
from scipy.stats.distributions import norm
from sobol import sobol_seq

from yaocptool.modelling import OptimalControlProblem, SystemModel, StochasticOCP
from yaocptool.stochastic import sample_parameter_normal_distribution_with_sobol


class PCEConverter:
    def __init__(self, socp, **kwargs):
        """

        :param StochasticOCP socp: Stochastic Optimal Control Problem
        :param int n_samples: number of samples of the parameters. If none is provided, the minimum number of samples
        will be used, depending on the number of uncertain parameters and polynomial order
        :param int pc_order: order of the polynomial, for the polynomial approximation. (default: 3)

        """
        self.socp = socp
        self.n_samples = None
        self.pc_order = 3
        self.lamb = 0.0

        self.variable_type = 'theta'
        self.stochastic_variables = []

        self.model = None  # type: SystemModel
        self.problem = None  # type: OptimalControlProblem
        self.sampled_parameters = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.n_samples is None:
            self.n_samples = self.n_pol_parameters

    @property
    def n_uncertain(self):
        return self.socp.n_p_unc + self.socp.n_uncertain_initial_condition

    @property
    def n_pol_parameters(self):
        n_pol_parameters = factorial(self.n_uncertain + self.pc_order) / (
                factorial(self.n_uncertain) * factorial(self.pc_order))
        return n_pol_parameters

    def _sample_parameters(self):
        n_samples = self.n_samples
        n_uncertain = self.n_uncertain

        mean = vertcat(self.socp.p_unc_mean, self.socp.uncertain_initial_conditions_mean)

        if self.socp.n_p_unc > 0 and self.socp.n_uncertain_initial_condition > 0:
            covariance = diagcat(self.socp.p_unc_cov, self.socp.uncertain_initial_conditions_cov)
        elif self.socp.n_p_unc > 0:
            covariance = self.socp.p_unc_cov
        elif self.socp.n_uncertain_initial_condition > 0:
            covariance = self.socp.uncertain_initial_conditions_cov
        else:
            raise ValueError("No uncertanties found n_p_unc = {}, "
                             "n_uncertain_initial_condition={}".format(self.socp.n_p_unc,
                                                                       self.socp.n_uncertain_initial_condition))

        dist = self.socp.p_unc_dist + self.socp.uncertain_initial_conditions_distribution

        for d in dist:
            if not d == 'normal':
                raise NotImplementedError('Distribution "{}" not implemented, only "normal" is available.'.format(d))

        sampled_epsilon = sample_parameter_normal_distribution_with_sobol(DM.zeros(mean.shape),
                                                                          DM.eye(covariance.shape[0]),
                                                                          n_samples)
        sampled_parameters = SX.zeros(n_uncertain, n_samples)
        for s in range(n_samples):
            sampled_parameters[:, s] = mean + mtimes(sampled_epsilon[:, s].T, chol(covariance)).T
        return sampled_parameters

    def _create_cost_ode_of_sample(self, model_s):
        cost = self.socp.L

        original_vars = vertcat(self.socp.model.x_sym, self.socp.model.y_sym,
                                self.socp.model.u_sym, self.socp.model.p_sym,
                                self.socp.model.theta_sym, self.socp.model.t_sym,
                                self.socp.model.tau_sym, self.socp.model.x_0_sym)

        new_vars = vertcat(model_s.x_sym, model_s.y_sym,
                           model_s.u_sym, model_s.p_sym,
                           model_s.theta_sym, model_s.t_sym,
                           model_s.tau_sym, model_s.x_0_sym)

        cost = substitute(cost, original_vars, new_vars)

        return cost

    def convert_socp_to_ocp_with_pce(self):
        # Sample Parameters
        self.sampled_parameters = self._sample_parameters()

        # Build the model
        self.model, cost_list = self._create_model(self.sampled_parameters)

        # Build the problem
        self.problem = self._create_problem(self.model, self.sampled_parameters)

        # Get PCE parameters
        ls_factor, exp_phi, psi_fcn = self._get_ls_factor()

        # Include the stochastic objective function
        self._construct_stochastic_objective(cost_list, exp_phi, ls_factor, self.problem)

        # uncertain constraints
        self._include_statistics_eqs_of_stochastics_variables(exp_phi, ls_factor, self.model, self.problem)

        return self.problem

    def _include_statistics_eqs_of_stochastics_variables(self, exp_phi, ls_factor, model, problem):
        self.stochastic_variables = vertcat(*self.stochastic_variables)

        for i in range(self.stochastic_variables.numel()):
            var = self.stochastic_variables[i]
            if var.is_symbolic():
                name = var.name()
            else:
                name = 'stoch_var_' + str(i)

            _, _, _ = self._include_statistics_of_expression(var, name, exp_phi, ls_factor, model, problem)

        for i in range(self.socp.n_g_stochastic):
            var = self.socp.g_stochastic_ineq[i]
            rhs = self.socp.g_stochastic_rhs[i]

            name = 'stoch_constr_' + str(i)
            [stoch_ineq_mean, stoch_ineq_var, _] = self._include_statistics_of_expression(var, name, exp_phi, ls_factor,
                                                                                          model, problem)

            stoch_cosntr_viol_prob = problem.create_optimization_theta('viol_prob_' + name, new_theta_opt_max=0.0)
            k_viol = sqrt(self.socp.g_stochastic_prob[i] / (1 - self.socp.g_stochastic_prob[i]))

            problem.include_time_equality(stoch_cosntr_viol_prob
                                          - (k_viol * sqrt(fmax(1e-6, stoch_ineq_var)) + stoch_ineq_mean - rhs),
                                          when='end')

    def _include_statistics_of_expression(self, expr, name, exp_phi, ls_factor, model, problem):
        if self.variable_type == 'algebraic':
            raise NotImplementedError("stochastic variables as algebraic variables are not implemented")

        if self.variable_type == 'theta':
            var_vector = self._get_expression_in_each_scenario(expr, model)

            stochastic_var_mean = problem.create_optimization_theta(name + '_mean', 1)
            stochastic_var_var = problem.create_optimization_theta(name + '_var', 1)
            stochastic_var_par = problem.create_optimization_theta(name + '_par', self.n_pol_parameters)

            problem.include_time_equality(stochastic_var_par - mtimes(ls_factor, var_vector), when='end')
            problem.include_time_equality(stochastic_var_mean - stochastic_var_par[0], when='end')
            problem.include_time_equality(
                stochastic_var_var - mtimes((stochastic_var_par[1:] ** 2).T, exp_phi[1:]), when='end')

            return stochastic_var_mean, stochastic_var_var, stochastic_var_par

    def _construct_stochastic_objective(self, cost_list, exp_phi, ls_factor, problem):
        # Exp(J) and Var(J)
        cost_pol_par = problem.create_optimization_parameter('cost_pol_par', self.n_pol_parameters)
        mean_cost = problem.create_optimization_parameter('cost_mean', 1)
        var_cost = problem.create_optimization_parameter('cost_var', 1)
        problem.include_final_time_equality(cost_pol_par - mtimes(ls_factor, cost_list))
        problem.include_final_time_equality(var_cost - mtimes((cost_pol_par[1:] ** 2).T, exp_phi[1:]))
        problem.include_final_time_equality(mean_cost - cost_pol_par[0])
        problem.V = cost_pol_par[0]

    def _create_problem(self, model, sampled_parameter):
        # Problem
        problem = OptimalControlProblem(model)
        problem.name = self.socp.name + '_PCE'

        problem.p_opt = substitute(self.socp.p_opt, self.socp.get_p_without_p_unc(), problem.model.p_sym)
        problem.theta_opt = substitute(self.socp.theta_opt, self.socp.model.theta_sym, problem.model.theta_sym)

        problem.x_max = repmat(vertcat(self.socp.x_max, inf), self.n_samples)
        problem.y_max = repmat(self.socp.y_max, self.n_samples)
        problem.u_max = self.socp.u_max
        problem.delta_u_max = self.socp.delta_u_max
        problem.p_opt_max = self.socp.p_opt_max
        problem.theta_opt_max = self.socp.theta_opt_max

        problem.x_min = repmat(vertcat(self.socp.x_min, -inf), self.n_samples)
        problem.y_min = repmat(self.socp.y_min, self.n_samples)
        problem.u_min = self.socp.u_min
        problem.delta_u_min = self.socp.delta_u_min
        problem.p_opt_min = self.socp.p_opt_min
        problem.theta_opt_min = self.socp.theta_opt_min

        problem.t_f = self.socp.t_f

        if depends_on(self.socp.g_eq, self.socp.model.x_sym) or depends_on(self.socp.g_eq, self.socp.model.y_sym):
            raise NotImplementedError('Case where "g_eq" depends on "model.x_sym" or "model.y_sym" is not implemented ')

        if depends_on(self.socp.g_ineq, self.socp.model.x_sym) or depends_on(self.socp.g_ineq, self.socp.model.y_sym):
            raise NotImplementedError('Case where "g_ineq" depends on "model.x_sym" '
                                      'or "model.y_sym" is not implemented ')

        original_vars = vertcat(self.socp.model.u_sym, self.socp.get_p_without_p_unc(),
                                self.socp.model.theta_sym, self.socp.model.t_sym,
                                self.socp.model.tau_sym)

        new_vars = vertcat(problem.model.u_sym, problem.model.p_sym,
                           problem.model.theta_sym, problem.model.t_sym,
                           problem.model.tau_sym)

        if not self.socp.n_h_initial == self.socp.model.n_x:
            problem.h_initial = vertcat(problem.h_initial, substitute(self.socp.h_initial[:self.socp.model.n_x],
                                                                      original_vars, new_vars))
        problem.h_final = substitute(self.socp.h_final, original_vars, new_vars)

        problem.g_eq = substitute(self.socp.g_eq, original_vars, new_vars)
        problem.g_ineq = substitute(self.socp.g_ineq, original_vars, new_vars)
        problem.time_g_eq = self.socp.time_g_eq
        problem.time_g_ineq = self.socp.time_g_ineq

        for i in range(self.socp.n_uncertain_initial_condition):
            ind = self.socp.get_uncertain_initial_cond_indices()[i]
            x_ind_s = problem.model.x_0_sym[ind::(self.socp.model.n_x + 1)]
            problem.h_initial = substitute(problem.h_initial, x_ind_s, sampled_parameter[self.socp.n_p_unc + i, :].T)
            problem.h_final = substitute(problem.h_final, x_ind_s, sampled_parameter[self.socp.n_p_unc + i, :].T)

            problem.g_eq = substitute(problem.g_eq, x_ind_s, sampled_parameter[self.socp.n_p_unc + i, :].T)
            problem.g_ineq = substitute(problem.g_ineq, x_ind_s, sampled_parameter[self.socp.n_p_unc + i, :].T)

        problem.last_u = self.socp.last_u

        problem.y_guess = repmat(self.socp.y_guess, self.n_samples) if self.socp.y_guess is not None else None
        problem.u_guess = self.socp.u_guess
        problem.x_0 = repmat(vertcat(self.socp.x_0, 0), self.n_samples) if self.socp.x_0 is not None else None

        problem.parametrized_control = self.socp.parametrized_control
        problem.positive_objective = self.socp.parametrized_control
        problem.NULL_OBJ = self.socp.NULL_OBJ

        if not is_equal(self.socp.S, 0) or not is_equal(self.socp.V, 0):
            raise NotImplementedError

        return problem

    def _create_model(self, sampled_parameters):
        sampled_parameters_p_unc = sampled_parameters[:self.socp.n_p_unc, :]
        model = SystemModel(name=self.socp.model.name + '_PCE')

        model.include_control(self.socp.model.u_sym)
        model.include_parameter(self.socp.get_p_without_p_unc())
        model.include_theta(self.socp.model.theta_sym)

        u_global = model.u_sym
        p_global = model.p_sym
        theta_global = model.theta_sym

        t_global = model.t_sym
        tau_global = model.tau_sym

        cost_list = []
        for s in range(self.n_samples):
            model_s = self.socp.model.get_hardcopy()

            # cost of sample
            cost_ode = self._create_cost_ode_of_sample(model_s)
            cost_s = model_s.create_state('cost_' + str(s))
            model_s.include_system_equations(ode=cost_ode)

            # replace the parameter variable with the sampled variable
            p_unc_s = model_s.p_sym.get(False, self.socp.get_p_unc_indices())

            model_s.replace_variable(p_unc_s, sampled_parameters_p_unc[:, s])
            model_s.remove_parameter(p_unc_s)

            # replace the model variables with the global variables
            model_s.replace_variable(model_s.u_sym, u_global)
            model_s.replace_variable(model_s.p_sym, p_global)
            model_s.replace_variable(model_s.theta_sym, theta_global)
            model_s.remove_control(model_s.u_sym)
            model_s.remove_parameter(model_s.p_sym)
            model_s.remove_theta(model_s.theta_sym)
            model_s.t_sym = t_global
            model_s.tau_sym = tau_global

            # merge the sample model in the unique model
            model.merge(model_s)

            # collect the sample cost
            cost_list.append(cost_s)

        cost_list = vertcat(*cost_list)
        return model, cost_list

    def _get_ls_factor(self):
        return get_ls_factor(self.n_uncertain, self.n_samples, self.pc_order, self.lamb)

    def _get_expression_in_each_scenario(self, expr, model):
        """

        :param SystemModel model:
        """
        expr_list = []
        for s in range(self.n_samples):
            x_s = model.x_sym[s * (self.socp.model.n_x + 1):(s + 1) * (self.socp.model.n_x + 1)][:self.socp.model.n_x]
            x_0_s = model.x_0_sym[s * (self.socp.model.n_x + 1):
                                  (s + 1) * (self.socp.model.n_x + 1)][:self.socp.model.n_x]
            y_s = model.y_sym[s * self.socp.model.n_y:(s + 1) * self.socp.model.n_y]

            original_vars = vertcat(self.socp.model.x_sym, self.socp.model.y_sym,
                                    self.socp.model.u_sym, self.socp.get_p_without_p_unc(),
                                    self.socp.model.theta_sym, self.socp.model.t_sym,
                                    self.socp.model.tau_sym, self.socp.model.x_0_sym, self.socp.p_unc)

            new_vars = vertcat(x_s, y_s,
                               model.u_sym, model.p_sym,
                               model.theta_sym, model.t_sym,
                               model.tau_sym, x_0_s, self.sampled_parameters[:, s])

            new_exp = substitute(expr, original_vars, new_vars)
            expr_list.append(new_exp)
        return vertcat(*expr_list)


def get_ls_factor(n_uncertain, n_samples, pc_order, lamb=0.0):
    # Uncertain parameter design
    sobol_design = sobol_seq.i4_sobol_generate(n_uncertain, n_samples, ceil(np.log2(n_samples)))
    sobol_samples = np.transpose(sobol_design)
    for i in range(n_uncertain):
        sobol_samples[:, i] = norm(loc=0., scale=1.).ppf(sobol_samples[:, i])

    # Polynomial function definition
    x = SX.sym('x')
    he0fcn = Function('He0fcn', [x], [1.])
    he1fcn = Function('He1fcn', [x], [x])
    he2fcn = Function('He2fcn', [x], [x ** 2 - 1])
    he3fcn = Function('He3fcn', [x], [x ** 3 - 3 * x])
    he4fcn = Function('He4fcn', [x], [x ** 4 - 6 * x ** 2 + 3])
    he5fcn = Function('He5fcn', [x], [x ** 5 - 10 * x ** 3 + 15 * x])
    he6fcn = Function('He6fcn', [x], [x ** 6 - 15 * x ** 4 + 45 * x ** 2 - 15])
    he7fcn = Function('He7fcn', [x], [x ** 7 - 21 * x ** 5 + 105 * x ** 3 - 105 * x])
    he8fcn = Function('He8fcn', [x], [x ** 8 - 28 * x ** 6 + 210 * x ** 4 - 420 * x ** 2 + 105])
    he9fcn = Function('He9fcn', [x], [x ** 9 - 36 * x ** 7 + 378 * x ** 5 - 1260 * x ** 3 + 945 * x])
    he10fcn = Function('He10fcn', [x], [x ** 10 - 45 * x ** 8 + 640 * x ** 6 - 3150 * x ** 4 + 4725 * x ** 2 - 945])
    helist = [he0fcn, he1fcn, he2fcn, he3fcn, he4fcn, he5fcn, he6fcn, he7fcn, he8fcn, he9fcn, he10fcn]

    # Calculation of factor for least-squares
    xu = SX.sym("xu", n_uncertain)
    exps = (p for p in product(range(pc_order + 1), repeat=n_uncertain) if sum(p) <= pc_order)
    exps.next()
    exps = list(exps)

    psi = SX.ones(factorial(n_uncertain + pc_order) / (factorial(n_uncertain) * factorial(pc_order)))
    for i in range(len(exps)):
        for j in range(n_uncertain):
            psi[i + 1] *= helist[exps[i][j]](xu[j])
    psi_fcn = Function('PSIfcn', [xu], [psi])

    nparameter = SX.size(psi)[0]
    psi_matrix = SX.zeros(n_samples, nparameter)
    for i in range(n_samples):
        psi_a = psi_fcn(sobol_samples[i, :])
        for j in range(SX.size(psi)[0]):
            psi_matrix[i, j] = psi_a[j]

    psi_t_psi = mtimes(psi_matrix.T, psi_matrix) + lamb * DM.eye(nparameter)
    chol_psi_t_psi = chol(psi_t_psi)
    inv_chol_psi_t_psi = solve(chol_psi_t_psi, SX.eye(nparameter))
    inv_psi_t_psi = mtimes(inv_chol_psi_t_psi, inv_chol_psi_t_psi.T)

    ls_factor = mtimes(inv_psi_t_psi, psi_matrix.T)
    ls_factor = DM(ls_factor)

    # Calculation of expectations for variance function
    n_sample_expectation_vector = 100000
    x_sample = np.random.multivariate_normal(np.zeros(n_uncertain), np.eye(n_uncertain), n_sample_expectation_vector)
    psi_squared_sum = DM.zeros(SX.size(psi)[0])
    for i in range(n_sample_expectation_vector):
        psi_squared_sum += psi_fcn(x_sample[i, :]) ** 2
    expectation_vector = psi_squared_sum / n_sample_expectation_vector

    return ls_factor, expectation_vector, psi_fcn
