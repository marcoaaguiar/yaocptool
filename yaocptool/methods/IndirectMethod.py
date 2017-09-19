# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:52 2016

@author: marco
"""
import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX, MX, DM, inf, repmat, vertcat, collocation_points, \
                    substitute, linspace, integrator, vec, nlpsol, \
                    dot, gradient, hessian, mtimes, inv, fmin, fmax, Function
  
from solutionmethods import *
import warnings

import matplotlib.pyplot as plt


class IndirectMethod(SolutionMethodsBase):
    def __init__(self, problem, **kwargs):
        SolutionMethodsBase.__init__(self, **kwargs)

        self.problem = problem
        self.solution_class = 'indirect'

        self.parametrized_control = False
        self.hasCostState = False
        
        self.checkBounds()
        
    def checkBounds(self):
        for i in range(self.model.Nx):
            if not self.problem.x_min[i] == -inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.x_min[i] = -inf
                
            if not self.problem.x_max[i] == inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.x_max[i] = inf
                
        for i in range(self.model.Ny):
            if not self.problem.y_min[i] == -inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.y_min[i] = -inf
                
            if not self.problem.y_max[i] == inf:
                warnings.warn('Problem contains state constraints, they will be ignored')
                self.problem.y_max[i] = inf
                
    def calculateOptimalControl(self):
        ddH_dudu, dH_du = hessian(self.problem.H,self.model.u_sym)
        u_opt = -mtimes(inv(ddH_dudu),substitute(dH_du, self.model.u_sym,0))

        for i in range(self.model.Nu):
            if not self.problem.u_min[i] == -inf:
                u_opt[i] = fmax(u_opt[i], self.problem.u_min[i])
                
            if not self.problem.u_max[i] == inf:
                u_opt[i] = fmin(u_opt[i], self.problem.u_max[i])
        return u_opt

    def ReplaceWithOptimalControl(self, u_opt):
        self.problem.replaceVariable(self.model.u_sym, u_opt, 'u')
        self.model.control_function = u_opt
        self.problem.removeControl(self.model.u_sym)
    
    def prepare(self):
        self.includeAdjointStates()
        self.u_opt = self.calculateOptimalControl()
        self.ReplaceWithOptimalControl(self.u_opt)
        

        