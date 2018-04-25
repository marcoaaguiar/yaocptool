from casadi import vertcat, DM

from yaocptool.modelling import OptimalControlProblem


class StochasticOCP(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.g_stochastic_ineq = vertcat([])
        self.g_stochastic_rhs = vertcat([])
        self.g_stochastic_prob = vertcat([])

        OptimalControlProblem.__init__(self, model, **kwargs)

    def include_time_chance_inequality(self, ineq, prob, rhs=None, when='default'):
        """Include time dependent chance inequality.
        Prob[ineq(..., t) <= rhs] <= prob, for t \in [t_0, t_f]

        The inequality is concatenated to "g_stochastic_ineq"

        :param ineq: inequality
        :param rhs: Right-hand size of the inequality
        :param prob: Chance/probability of the constraint being satisfied
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
        self.g_stochastic_prob = vertcat(self.g_stochastic_prob, ineq)
        self.time_g_ineq.extend([when] * ineq.numel())
