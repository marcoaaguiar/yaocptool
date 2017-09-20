# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:40:15 2016

@author: marco
"""
import sys

sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
# if not 'casadi' in sys.modules:
from casadi import SX, MX, DM, inf, repmat, vertcat, collocation_points, \
    substitute, linspace, integrator, horzcat, IM, vec
import matplotlib.pyplot as plt

import solutionmethods
from yaocptool.modelling_classes.ocp import OptimalControlProblem


class DirectMethod(solutionmethods.SolutionMethodsBase):
    def __init__(self, problem, **kwargs):

        """

        :type problem: OptimalControlProblem
        """
        solutionmethods.SolutionMethodsBase.__init__(self, problem, **kwargs)

        self.solution_class = 'direct'

        self.parametrized_control = False
        self.hasCostState = False

    def parametrizeControl(self):
        u_pol = self.createControlApproximation()

        self.problem.replaceVariable(self.model.u_sym, u_pol)

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
        self.parametrizeControl()
        self.createCostState()
        self.problem.makeFinalCostFunction()
