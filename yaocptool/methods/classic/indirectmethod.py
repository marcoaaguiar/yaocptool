# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:52 2016

@author: marco
"""
import warnings
from yaocptool.modelling.ocp import OptimalControlProblem

from casadi import inf, substitute, hessian, inv, fmin, fmax, is_equal, mtimes, DM

from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


class IndirectMethod(SolutionMethodsBase):
    def __init__(
        self, problem: OptimalControlProblem, create_cost_state: bool = False, **kwargs
    ):
        """
        :param problem: yaocptool.modelling.ocp.OptimalControlProblem
        :param create_cost_state: If True a cost state will be created to keep track of the dynamic cost.
        :param integrator_type: str
        :param solution_method: str
        :param degree: int
        :param discretization_scheme: str 'multiple-shooting' | 'collocation'
        """
        self.create_cost_state = create_cost_state
        self.has_cost_state = False

        super().__init__(problem, **kwargs)

        self.solution_class = "indirect"
        self.initial_guess_heuristic = "problem_info"

        self._check_bounds()

    @property
    def degree_control(self) -> int:
        return self.degree

    @degree_control.setter
    def degree_control(self, value: int):
        raise ValueError(
            "Cannot set degree_control of indirect method, use degree instead"
        )

    def prepare(self):
        super(IndirectMethod, self).prepare()
        self.problem.create_adjoint_states()
        if self.create_cost_state:
            self.has_cost_state = True
            self.problem.create_cost_state()
        u_opt = self.calculate_optimal_control()
        self.replace_with_optimal_control(u_opt)

    def _check_bounds(self):
        for i in range(self.model.n_x):
            if self.problem.x_min[i] != -inf:
                warnings.warn(
                    "Problem contains state constraints, they will be ignored"
                )
                self.problem.x_min[i] = -inf

            if self.problem.x_max[i] != inf:
                warnings.warn(
                    "Problem contains state constraints, they will be ignored"
                )
                self.problem.x_max[i] = inf

        for i in range(self.model.n_y):
            if self.problem.y_min[i] != -inf:
                warnings.warn(
                    "Problem contains algebraic constraints, they will be ignored"
                )
                self.problem.y_min[i] = -inf

            if self.problem.y_max[i] != inf:
                warnings.warn(
                    "Problem contains algebraic constraints, they will be ignored"
                )
                self.problem.y_max[i] = inf

    def calculate_optimal_control(self):
        dd_h_dudu, d_h_du = hessian(self.problem.H, self.model.u)
        if is_equal(dd_h_dudu, DM.zeros(self.model.n_u, self.model.n_u)):
            # TODO: Implement the case where the controls are linear on the Hamiltonian ("Bang-Bang" control)
            raise Exception(
                'The Hamiltonian "H" is not strictly convex with respect to the control "u". '
                + "The obtained hessian d^2 H/du^2 = 0"
            )
        # if not ddH_dudu.is_constant():
        #     raise NotImplementedError('The Hessian of the Hamiltonian with respect to "u" is not constant,
        #                                this case has not been implemented')

        u_opt = -mtimes(inv(dd_h_dudu), substitute(d_h_du, self.model.u, 0.0))

        for i in range(self.model.n_u):
            if not self.problem.u_min[i] == -inf:
                u_opt[i] = fmax(u_opt[i], self.problem.u_min[i])

            if not self.problem.u_max[i] == inf:
                u_opt[i] = fmin(u_opt[i], self.problem.u_max[i])
        return u_opt

    def replace_with_optimal_control(self, u_opt):
        self.problem.parametrize_control(self.model.u, u_opt)
