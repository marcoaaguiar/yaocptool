# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:52 2016

@author: marco
"""
import warnings

from casadi import inf, substitute, hessian, inv, fmin, fmax, is_equal, mtimes, DM, SX, dot, gradient, vertcat, jacobian

from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


class IndirectMethod(SolutionMethodsBase):
    def __init__(self, problem, **kwargs):
        """
        :param problem: yaocptool.modelling.ocp.OptimalControlProblem
        :param integrator_type: str
        :param solution_method: str
        :param degree: int
        :param discretization_scheme: str 'multiple-shooting' | 'collocation'
        """
        self.hasCostState = False

        super(IndirectMethod, self).__init__(problem, **kwargs)

        self.solution_class = 'indirect'
        self.initial_guess_heuristic = 'problem_info'

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

    def include_adjoint_states(self):
        lamb = SX.sym('lamb', self.model.n_x)
        nu = SX.sym('nu', self.model.n_y)

        self.problem.eta = SX.sym('eta', self.problem.n_h_final)

        self.problem.H = self.problem.L + dot(lamb, self.model.ode) + dot(nu, self.model.alg)

        l_dot = -gradient(self.problem.H, self.model.x_sym)
        alg_eq = gradient(self.problem.H, self.model.y_sym)

        self.problem.include_state(lamb, l_dot, suppress=True)
        self.model.has_adjoint_variables = True

        self.problem.include_algebraic(nu, alg_eq)

        self.problem.h_final = vertcat(self.problem.h_final,
                                       self.model.lamb_sym - gradient(self.problem.V, self.model.x_sys_sym)
                                       - mtimes(jacobian(self.problem.h_final, self.model.x_sys_sym).T,
                                                self.problem.eta))

    def calculate_optimal_control(self):
        dd_h_dudu, d_h_du = hessian(self.problem.H, self.model.u_sym)
        if is_equal(dd_h_dudu, DM.zeros(self.model.n_u, self.model.n_u)):
            # TODO: Implement the case where the controls are linear on the Hamiltonian ("Bang-Bang" control)
            raise Exception('The Hamiltonian "H" is not strictly convex with respect to the control "u". '
                            + 'The obtained hessian d^2 H/du^2 = 0')
        # if not ddH_dudu.is_constant():
        #     raise NotImplementedError('The Hessian of the Hamiltonian with respect to "u" is not constant,
        #                                this case has not been implemented')

        u_opt = -mtimes(inv(dd_h_dudu), substitute(d_h_du, self.model.u_sym, 0))

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
