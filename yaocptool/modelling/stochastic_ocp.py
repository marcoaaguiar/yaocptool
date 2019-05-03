from casadi import vertcat, DM, diagcat

from yaocptool import find_variables_indices_in_vector, remove_variables_from_vector_by_indices
from yaocptool.modelling import OptimalControlProblem


class StochasticOCP(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.g_stochastic_ineq = vertcat([])
        self.g_stochastic_rhs = vertcat([])
        self.g_stochastic_prob = vertcat([])

        self.p_unc = vertcat([])
        self.p_unc_mean = vertcat([])
        self.p_unc_cov = vertcat([])
        self.p_unc_dist = []

        self.uncertain_initial_conditions = vertcat([])
        self.uncertain_initial_conditions_mean = vertcat([])
        self.uncertain_initial_conditions_cov = vertcat([])
        self.uncertain_initial_conditions_distribution = []

        OptimalControlProblem.__init__(self, model, **kwargs)

    @property
    def n_p_unc(self):
        return self.p_unc.numel()

    @property
    def n_g_stochastic(self):
        return self.g_stochastic_ineq.numel()

    @property
    def n_uncertain_initial_condition(self):
        return self.uncertain_initial_conditions.numel()

    def create_uncertain_parameter(self, name, mean, var, size=1, distribution='normal'):
        par = self.model.create_parameter(name=name, size=size)

        self.set_parameter_as_uncertain_parameter(par, mean, var, distribution=distribution)
        return par

    def include_uncertain_parameter(self, par, mean, cov, distribution='normal'):
        self.model.include_parameter(par)
        self.set_parameter_as_uncertain_parameter(par, mean, cov, distribution=distribution)

    def set_parameter_as_uncertain_parameter(self, par, mean, cov, distribution='normal'):
        par = vertcat(par)
        mean = vertcat(mean)
        cov = vertcat(cov)

        if not par.numel() == mean.numel():
            raise ValueError('Size of "par" and "mean" differ. par.numel()={} '
                             'and mean.numel()={}'.format(par.numel(), mean.numel()))
        if not cov.shape == (mean.numel(), mean.numel()):
            raise ValueError('The input "cov" is not a square matrix of same size as "mean". '
                             'cov.shape={} and mean.numel()={}'.format(cov.shape, mean.numel()))

        self.p_unc = vertcat(self.p_unc, par)
        self.p_unc_mean = vertcat(self.p_unc_mean, mean)
        # TODO: Work around for casadi diagcat bug, remove when patched.
        if self.p_unc_cov.numel() == 0:
            self.p_unc_cov = cov
        else:
            self.p_unc_cov = diagcat(self.p_unc_cov, cov)
        self.p_unc_dist = self.p_unc_dist + [distribution] * par.numel()

    def set_initial_condition_as_uncertain(self, par, mean, cov, distribution='normal'):
        par = vertcat(par)
        mean = vertcat(mean)
        cov = vertcat(cov)

        if not par.numel() == mean.numel():
            raise ValueError('Size of "par" and "mean" differ. par.numel()={} '
                             'and mean.numel()={}'.format(par.numel(), mean.numel()))
        if not cov.shape == (mean.numel(), mean.numel()):
            raise ValueError('The input "cov" is not a square matrix of same size as "mean". '
                             'cov.shape={} and mean.numel()={}'.format(cov.shape, mean.numel()))

        self.uncertain_initial_conditions = vertcat(self.uncertain_initial_conditions, par)
        self.uncertain_initial_conditions_mean = vertcat(self.uncertain_initial_conditions_mean, mean)
        self.uncertain_initial_conditions_distribution = self.uncertain_initial_conditions_distribution + [
            distribution] * par.numel()
        if cov.numel() > 0 and self.uncertain_initial_conditions_cov.numel() > 0:
            self.uncertain_initial_conditions_cov = diagcat(self.uncertain_initial_conditions_cov, cov)
        elif cov.numel() > 0:
            self.uncertain_initial_conditions_cov = cov
        else:
            self.uncertain_initial_conditions_cov = self.uncertain_initial_conditions_cov

    def get_p_without_p_unc(self):
        return remove_variables_from_vector_by_indices(self.get_p_unc_indices(), self.model.p_sym)

    def get_p_unc_indices(self):
        return find_variables_indices_in_vector(self.p_unc, self.model.p_sym)

    def get_uncertain_initial_cond_indices(self):
        return find_variables_indices_in_vector(self.uncertain_initial_conditions, self.model.x_0_sym)

    def include_time_chance_inequality(self, ineq, prob, rhs=None, when='default'):
        r"""Include time dependent chance inequality.
        Prob[ineq(..., t) <= rhs] <= prob, for t \in [t_0, t_f]

        The inequality is concatenated to "g_stochastic_ineq"

        :param ineq: inequality
        :param list|DM rhs: Right-hand size of the inequality
        :param list|DM prob: Chance/probability of the constraint being satisfied
        :param str when: Can be 'default', 'end', 'start'.
            'start' - the constraint will be evaluated at the start of every finite element
            'end' - the constraint will be evaluated at the end of every finite element
            'default' - will be evaluated at each collocation point of every finite element.
            For the multiple shooting, the constraint will be evaluated at the end of each
            finite element
        """
        if isinstance(ineq, list):
            ineq = vertcat(*ineq)
        if isinstance(rhs, list):
            rhs = vertcat(*rhs)
        if isinstance(prob, list):
            prob = vertcat(*prob)

        if isinstance(rhs, (float, int)):
            rhs = vertcat(rhs)

        if isinstance(prob, (float, int)):
            prob = vertcat(prob)

        if rhs is None:
            rhs = DM.zeros(ineq.shape)

        if not ineq.numel() == rhs.numel():
            raise ValueError('Given inequality does not have the same dimensions of the provided "rhs". '
                             'Size of ineq: {}, size of rhs: {}'.format(ineq.shape, rhs.shape))
        if not ineq.numel() == rhs.numel():
            raise ValueError('Given inequality does not have the same dimensions of the provided "prob". '
                             'Size of ineq: {}, size of prob: {}'.format(ineq.shape, prob.shape))

        self.g_stochastic_ineq = vertcat(self.g_stochastic_ineq, ineq)
        self.g_stochastic_rhs = vertcat(self.g_stochastic_rhs, rhs)
        self.g_stochastic_prob = vertcat(self.g_stochastic_prob, prob)
