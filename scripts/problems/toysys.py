# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX,DM, inf, repmat, vertcat, collocation_points, DM, \
                    substitute, cos, sin, pi, diag, horzcat, jacobian, hessian
import matplotlib.pyplot as plt


import modelproblem


class ToySystem(modelproblem.SystemModel):
    def __init__(self, **kwargs):
        modelproblem.SystemModel.__init__(self, Nx =2, Ny= 0, Nu=1)
        
        x_sym = self.x_sym
        #y_sym = self.y_sym 
        u = self.u_sym
        
        #model extracte from Tracking trajectories of the cart-pendulum system
        # Mazenc

        
        for (k, v) in kwargs.items():
            exec(k + ' = ' + repr(v))
#        m = 0.853
#        M = 1
#        l = 0.323
        
        x_1 = x_sym[0]
        x_2 = x_sym[1]
        
        
        ode = vertcat(
            x_2,
            -x_1-x_2+u
        )
        self.includeSystemEquations(ode)


class ToyProblem(modelproblem.OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1,1]), 'R':.1, 'Qv':diag([1,1]), 'x_ref':DM([0,0])}  
#        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
        self.state_constraints = False
        self.control_constraints = False        
        
        modelproblem.OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5.
        self.x_0 = [0,1]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[1] = 2.
            self.x_min[1] = -2.
        if self.control_constraints:
            self.u_max[0] = 5.
            self.u_min[0] = -5.
        
if __name__ == '__main__':
    sys_model = ToySystem()
    
    prob = ToyProblem(sys_model)
