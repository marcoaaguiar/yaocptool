# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX,DM, inf, repmat, vertcat, collocation_points, \
                    substitute, cos, sin, pi, diag, horzcat, jacobian, hessian, dot
import matplotlib.pyplot as plt


from yaocptool.modelling_classes.model_classes import SystemModel, SuperModel
from yaocptool.modelling_classes.ocp import OptimalControlProblem, SuperOCP

if __name__ == '__main__':
    import solutionmethods, DirectMethod, IndirectMethod,InteriorPoint, AugmentedLagrange


class Distillation(SystemModel):
    def __init__(self):
        modelproblem.SystemModel.__init__(self, Nx =4, Ny= 2, Nu=1)
        
        x0= self.x_sym[0]
        x1= self.x_sym[1]
        x2= self.x_sym[2]
        x3= self.x_sym[3]
        
        z0 = self.y_sym[0]
        z1 = self.y_sym[1]
        u = self.u_sym
        # Parameters
        
        sF = 150
        mu_max = 0.9819
        nu_max = 2.3507
        K_s = 2.3349
        K_s_dash = 7.3097
        K_sI = 213.5899
        K_sI_dash = 5759.105
        K_p = 27.9036
        K_p_dash = 252.306
        K_pI = 41.2979
        K_pI_dash = 15.2430
        Y_ps = 0.4721

        ode = vertcat(z0 * x0 - u*x0/x3 , 
                      -z1*x0 / Y_ps + (u/x3)*(sF - x1), 
                     z1*x0 - (u/x3)*x2, 
                     u)
                     
        alg = vertcat((((mu_max*x1)/(K_s+x1+x1**2/K_sI))*(K_p)/(K_p+x2+x2**2/K_pI))-z0, 
               ((nu_max*x1)/(K_s_dash+x1+x1**2/K_sI_dash))*(K_p_dash)/(K_p_dash+x2+x2**2/K_pI_dash)-z1) 
        
        self.includeSystemEquations(ode, alg)


class ProdMaximization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
#        self.cost = {'Q': diag([1,1]), 'R':1 , 'x_ref':DM([0,0])}  
        self.state_constraints = False
        self.control_constraints = True        
        
        self.V = -model.x_sym[2]*model.x_sym[3]
#        self.L = -dot(model.ode[2],model.x_sym[3]) - dot(model.ode[3], model.x_sym[2])
        OptimalControlProblem.__init__(self, model)
        
        self.h_final = vertcat(self.model.x_sym[3]-5, self.model.x_sym[2] - 61)
        self.t_f  = 12
        self.x_0 = [0.03,  100,  0.4,  1.5]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[3] = 5 
        if self.control_constraints:
            self.u_max[0] = 1
            self.u_min[0] = 0


        
if __name__ == '__main__':
    model = Distillation()
    problem = ProdMaximization(model)
#    
#    indir_method = IndirectMethod.IndirectMethod(problem, degree = 1, finite_elements = 20, integrator_type = 'implicit')
#    x_sol, u_sol, V_sol = indir_method.solve()
#    x, u, t= indir_method.plotSimulate(x_sol, u_sol, [{'x':[0],'u':[0]},{'x':[2]}], 5)
    
    dir_method = DirectMethod.DirectMethod(problem, degree = 1, finite_elements = 48*2, integrator_type = 'implicit')
    x_sol, u_sol, V_sol = dir_method.solve()
    x, u, t= dir_method.plotSimulate(x_sol, u_sol, [{'x':[2]},{'x':[3]},{'u':[0]}], 10)

#    aug = AugmentedLagrange.AugmentedLagrange(problem, IndirectMethod.IndirectMethod, \
#        { 'integrator_type': 'explicit'},
#            max_iter = 6, mu_0 = 1, finite_elements = 50, degree = 5)
#    x_sol, u_sol =aug.solve()
    
#    x, u, t= aug.plotSimulate(x_sol, u_sol, [{'x':[2]},{'u':[0]}], 5)
