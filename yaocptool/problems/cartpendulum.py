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


from yaocptool.modelling_classes.model_classes import SystemModel
from yaocptool.modelling_classes.ocp import OptimalControlProblem


class PendulumCart(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, Nx =4, Ny= 0, Nu=1)
        
        x_sym = self.x_sym
        #y_sym = self.y_sym 
        u = self.u_sym
        
        #model extracte from Tracking trajectories of the cart-pendulum system
        # Mazenc

        g =  9.8
        l = 9.8/9.
        M = 1.
        m = 1.
        
        for (k, v) in kwargs.items():
            exec( k + ' = ' + `v`)
#        m = 0.853
#        M = 1
#        l = 0.323
        
        theta = x_sym[0]
        theta_dot = x_sym[1]
        
        x = x_sym[2]
        x_dot = x_sym[3]
        
        ode = vertcat(
            theta_dot,
            ((m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2)*cos(theta) +g*sin(theta))/l,
            x_dot,
            (m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2)
        )
        self.includeSystemEquations(ode)


class UpwardPendulumStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([10,0.1,0.1,0.1]), 'R':.1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
#        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False        
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5.
        self.x_0 = [pi/6., 0, 0,0]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2
        if self.state_constraints_2:
            self.x_max[3] = 3
            self.x_min[3] = -3
        if self.control_constraints:
            self.u_max[0] = 20.
            self.u_min[0] = -20.
            

class DownwardPendulumStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([10,0.1,0.1,0.1]), 'R':.1, 'Qv':diag([100,100,0,0]), 'x_ref':DM([pi,0,0,0])}  
        self.state_constraints = False
        self.control_constraints = False     

        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5
        self.x_0 = [pi/6, 0, 0,0]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[2] = 10
            self.x_min[2] = -10
        if self.control_constraints:
            self.u_max[0] = 2
            self.u_min[0] = -2
        
if __name__ == '__main__':
    sys_model = PendulumCart()
    
    prob = UpwardPendulumStabilization(sys_model)
