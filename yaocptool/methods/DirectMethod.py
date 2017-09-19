# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:40:15 2016

@author: marco
"""
import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX, MX, DM, inf, repmat, vertcat, collocation_points, \
                    substitute, linspace, integrator, horzcat, IM, vec
import matplotlib.pyplot as plt

import solutionmethods

class DirectMethod(solutionmethods.SolutionMethodsBase):
    def __init__(self, problem, **kwargs):
        # type: (object, object) -> object
        """w

        :rtype: object
        """
        solutionmethods.SolutionMethodsBase.__init__(self, **kwargs)

        self.problem = problem
        self.solution_class = 'direct'

        self.parametrized_control = False
        self.hasCostState = False


    def parametrizeControl(self):
        u_pol = self.createControlApproximation()
        
        self.model.ode = substitute(self.model.ode, self.model.u_sym, u_pol)
        self.model.alg = substitute(self.model.alg, self.model.u_sym, u_pol)
        self.model.alg_z = substitute(self.model.alg_z, self.model.u_sym, u_pol)
        self.model.con = substitute(self.model.con, self.model.u_sym, u_pol)
        self.problem.L = substitute(self.problem.L, self.model.u_sym, u_pol)
        
        if hasattr(self.problem, 'H'):
            self.problem.H = substitute(self.problem.H, self.model.u_sym, u_pol)
        if hasattr(self, 's'):
            self.s = substitute(self.s, self.model.u_sym, u_pol)
        #self.V = substitute(self.V, wm.u_sym, u_par)
        
        if 'g' in self.__dict__:
            g = self.g
            g = substitute(g, self.model.u_sym, u_pol)
            self.g = g
        return u_pol
                
    def prepare(self):
        # if self.discretization_method == 'multiple-shooting':
        self.parametrizeControl()
#        self.model.control_function = self.model.u_sym
        self.createCostState()
        self.problem.makeFinalCostFunction()
        
        
        
