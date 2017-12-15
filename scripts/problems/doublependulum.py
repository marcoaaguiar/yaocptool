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
                    mtimes, inv
import matplotlib.pyplot as plt


from yaocptool.modelling.model_classes import SystemModel, SuperModel
from yaocptool.modelling.ocp import OptimalControlProblem, SuperOCP

if __name__ == '__main__':
    import solutionmethods, DirectMethod, IndirectMethod,InteriorPoint, GradientMethod


class DoublePendulum(SystemModel):
    def __init__(self):
        SystemModel.__init__(self, n_x=6, n_y= 0, n_u=1)
        
        x_sym = self.x_sym
        #y_sym = self.y_sym 
        u = self.u_sym
        
        #model extracte from Tracking trajectories of the cart-pendulum system
        # Mazenc
        
        g =  9.8
        l_1 = 0.323
        l_2 = 0.480
        a_1 = 0.215
        a_2 = 0.223
        m_1 = 0.853
        m_2 = 0.510
        J_1 = 0.013
        J_2 = 0.019
        d_1 = 0.005
        d_2 = 0.005
        
        x_c = x_sym[0]
        dx_c = x_sym[1]
        phi_1 = x_sym[2]
        dphi_1 = x_sym[3]
        phi_2 = x_sym[4]
        dphi_2 = x_sym[5]
        
        M = SX(2,2)
        M[0,0] = J_1 + a_2**2*m_1 +l_1**2*m_2
        M[0,1] = M[1,0]= a_2*l_1*m_2*cos(phi_1 - phi_2)
        M[1,1] = J_2 + a_2**2*m_2
        c = SX(2,1)
        c[0] = d_1*dphi_1 + d_2*(dphi_1 - dphi_2) + l_1*m_2*a_2*sin(phi_1 - phi_2)*dphi_2**2 - (a_1*m_1 + l_1*m_2)*(g*sin(phi_1)+cos(phi_2)*u)        
        c[1] = d_2*(phi_2 - phi_1) - a_2*m_2*(g*sin(phi_2) + l_1*sin(phi_1 - phi_2)*dphi_1**2 + cos(phi_2)*u)
        
        ddphi = -mtimes(inv(M), c)
        ode = vertcat(
            dx_c,
            u,
            dphi_1,
            ddphi[0],
            dphi_2,
            ddphi[1]
        )
        self.include_system_equations(ode)


class UpwardStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
#        self.cost = {'Q': diag([10,0.1,0.1,0.1]), 'R':.1, 'Qv':diag([100,100,0,0]), 'x_ref':DM([0,0,0,0])}  
        self.cost = {'Q': diag([10, 0.1, 1, 0.1, 1, 0.1]), 'R':0.001, 'Qv':diag([1,1,1,1,1,1]), 'x_ref':DM([0,0,0,0,0,0])}  
        self.state_constraints = False
        self.control_constraints = False        
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 1.
        self.x_0 = [0, 0, pi/6.,0,pi/6,0]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[2] = 2.
            self.x_min[2] = -2.
        if self.control_constraints:
            self.u_max[0] = 6.
            self.u_min[0] = -6.
            

#class DownwardPendulumStabilization(modelproblem.OptimalControlProblem):
#    def __init__(self, model, **kwargs):
#        self.cost = {'Q': diag([10,0.1,0.1,0.1]), 'R':.1, 'Qv':diag([100,100,0,0]), 'x_ref':DM([pi,0,0,0])}  
#        self.state_constraints = False
#        self.control_constraints = False     
#
#        modelproblem.OptimalControlProblem.__init__(self, model, obj = self.cost)
#        
#        self.t_f  = 5
#        self.x_0 = [pi/6, 0, 0,0]
#
#        for (k, v) in kwargs.items():
#            setattr(self, k, v)
#            
#        if self.state_constraints:
#            self.x_max[2] = 10
#            self.x_min[2] = -10
#        if self.control_constraints:
#            self.u_max[0] = 2
#            self.u_min[0] = -2
#        
if __name__ == '__main__':
    model = DoublePendulum()
    problem = UpwardStabilization(model)
    
#    dir_method = DirectMethod.DirectMethod(problem, degree = 1, finite_elements = 300, integrator_type = 'explicit')
#    x_sol, u_sol, V_sol = dir_method.solve()
#    x, u, t= dir_method.plot_simulate(x_sol, u_sol, [{'x':[2,4]},{'u':[0]}], 1)

#    indir = IndirectMethod.IndirectMethod(problem, degree = 1, finite_elements = 80, integrator_type = 'implicit')
#    x_sol, u_sol, V_sol = indir.solve()
#    x, u, t= indir.plot_simulate(x_sol, u_sol, [{'x':[2,4]},{'u':[0]}], 1)

    grad = GradientMethod.GradientMethod(problem, degree = 1, max_iter =5, finite_elements = 40, integrator_type = 'implicit')
    x_sol, u_sol, V_sol = grad.solve()
    x, u, t= grad.plot_simulate(x_sol, u_sol, [{'x':[0]}, {'x':[1, 2, 3]}, {'u':[0]}], 10)