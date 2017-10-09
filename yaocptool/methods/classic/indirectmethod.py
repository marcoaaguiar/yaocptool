# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:52 2016

@author: marco
"""
from casadi import inf, substitute, hessian, inv, fmin, fmax, is_equal, mtimes, DM
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
import warnings
import yaocptool.modelling


class IndirectMethod(SolutionMethodsBase):
    def __init__(self, problem, **kwargs):
        """
        :param problem: yaocptool.modelling.ocp.OptimalControlProblem
        :param integrator_type: str
        :param solution_method: str
        :param degree: int
        :param discretization_scheme: str 'multiple-shooting' | 'collocation'
        """
        super(IndirectMethod, self).__init__(problem, **kwargs)

        self.problem = problem  # type: yaocptool.modelling.ocp.OptimalControlProblem
        self.solution_class = 'indirect'

        self.hasCostState = False

        self._check_bounds()

    def prepare(self):
        super(IndirectMethod, self).prepare()
        self.include_adjoint_states()
        u_opt = self.calculate_optimal_control()
        self.replace_with_optimal_control(u_opt)

    def _check_bounds(self):
        for i in range(self.model.n_x):
            if not self.problem.x_min[i] == -inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.x_min[i] = -inf

            if not self.problem.x_max[i] == inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.x_max[i] = inf

        for i in range(self.model.n_y):
            if not self.problem.y_min[i] == -inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.y_min[i] = -inf

            if not self.problem.y_max[i] == inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.y_max[i] = inf

    def calculate_optimal_control(self):
        ddH_dudu, dH_du = hessian(self.problem.H, self.model.u_sym)
        if is_equal(ddH_dudu, DM.zeros(self.model.n_u, self.model.n_u)):
            # TODO: Implement the case where the controls are linear on the Hamiltonina ("Bang-Bang" control)
            raise Exception('The Hamiltonian "H" is not strictly convex with respect to the control "u". '
                            + 'The obtained hessian d^2 H/du^2 = 0')
        # if not ddH_dudu.is_constant():
        #     raise NotImplementedError('The Hessian of the Hamiltonian with respect to "u" is not constant,
        #                                this case has not been implemented')

        u_opt = -mtimes(inv(ddH_dudu), substitute(dH_du, self.model.u_sym, 0))

        for i in range(self.model.n_u):
            if not self.problem.u_min[i] == -inf:
                u_opt[i] = fmax(u_opt[i], self.problem.u_min[i])

            if not self.problem.u_max[i] == inf:
                u_opt[i] = fmin(u_opt[i], self.problem.u_max[i])
        return u_opt

    def replace_with_optimal_control(self, u_opt):
        self.problem.replace_variable(self.model.u_sym, u_opt, 'u')
        self.model.u_func = u_opt
        self.problem.remove_control(self.model.u_sym)
