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
                    substitute, cos, sin, pi, diag, horzcat, jacobian, hessian, \
                    ones
import matplotlib.pyplot as plt


import modelproblem


class CSTRColumnSystem(modelproblem.SystemModel):
    def __init__(self, **kwargs):
        modelproblem.SystemModel.__init__(self, Nx =2, Ny= 1, Nu=3)
        
        x_sym = self.x_sym
        y_sym = self.y_sym 
        u_sym = self.u_sym
        


#==============================================================================
#         CSTR
#==============================================================================
        
        # Parameters
        
        zF0 = [0.9, 0.1]
        F0 = 460/60
        k1 = 0.341/60       
        F_default = 958/60
        
        for (k, v) in kwargs.items():
            exec( k + ' = ' + `v`)
        
        M_r = x_sym[0]
        M_rA = x_sym[1]
        
        z_FA = y_sym[0]
        
        D = u_sym[0]
        x_D = u_sym[1]
        F = u_sym[2]    
                
        Fs = F+F_default
        
        ode = vertcat(
            F0*zF0[0] + D*x_D - Fs*z_FA - k1*M_rA,
            F0 + D - Fs
        )
        alg = vertcat(
            z_FA -M_rA/M_r
        )
        
        
#==============================================================================
#         Column        
#==============================================================================
        
        # Parameters
        
        NT = 22     # Number of stages (including reboiler and total condenser: 
        NF = 13        # Location of feed stage (stages are counted from the bottom):
        alpha = DM([2, 1.0])        # Relative volatilities: alpha = [alpha1 alpha2...alphaNC]
        NC = length(alpha)        # Number of components
        # Parameters for Franci's Weir Formula L(i) = K*(Mi -Mow)^1.5
        Kuf = 1.625         # Constant above feed
        Kbf = 1.177         # Constant below feed
        
        # Need to change this value in case the flow rates change order of magnitude, Wier formula Li=Kf*(Mi-Muw)^1.5
        Muw = 4#0.25;             # Liquid holdup under weir (kmol)
        
        y = (x*diag(alpha[0:NC-1]))/((x*(alpha[0:NC-1] - 1).T + 1)*ones[0,NC-1])
        V = VB*ones(NT-1,1)
        V[NF:NT-1] = V[NF:NT-1] + (1-qF)*F

        L[1:NF,1] = Kbf*(real(M[2:NF] - Muw))**1.5         # Liquid flows below feed (Kmol/min)
        L[NF+1:NT-1,1] = Kuf*(real(M[NF+1:NT-1] - Muw))**1.5    # Liquid flows above feed (Kmol/min)
        L[NT-1,1]        = LT                               # Condenser's liquid flow (Kmol/min)
        
        self.includeSystemEquations(ode, alg)


class CSTRColumnProblem(modelproblem.OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1,1]), 'R':.1, 'x_ref':DM([2800,0])}  #'Qv':diag([1,1,1]), 
#        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
        self.state_constraints = False
        self.control_constraints = False        
        
        modelproblem.OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 100.
        self.x_0 = [2800, 2800*0.42855]

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
