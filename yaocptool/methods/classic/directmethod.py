# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:40:15 2016

@author: marco
"""
from casadi import substitute

from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase


class DirectMethod(SolutionMethodsBase):
    def __init__(self, problem, **kwargs):
        """
        :param problem: yaocptool.modelling.ocp.OptimalControlProblem
        :param integrator_type: str
        :param solution_method: str
        :param degree: int
        :param discretization_scheme: str 'multiple-shooting' | 'collocation'
        """
        super(DirectMethod, self).__init__(problem, **kwargs)

        self.solution_class = 'direct'

        self.hasCostState = False

    def _parametrize_control(self):
        u_pol = self.create_control_approximation()

        self.problem.replace_variable(self.model.u_sym, u_pol)

        if hasattr(self.problem, 'H'):
            self.problem.H = substitute(self.problem.H, self.model.u_sym, u_pol)
        if hasattr(self, 's'):
            self.s = substitute(self.s, self.model.u_sym, u_pol)

        if 'g' in self.__dict__:
            g = self.g
            g = substitute(g, self.model.u_sym, u_pol)
            self.g = g
        return u_pol

    def prepare(self):
        super(DirectMethod, self).prepare()
        self._parametrize_control()
        self._create_cost_state()
        # self.problem.make_final_cost_function()
