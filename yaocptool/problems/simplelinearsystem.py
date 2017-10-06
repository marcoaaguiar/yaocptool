# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import DM, vertcat, diag

from yaocptool.modelling.model_classes import SystemModel
from yaocptool.modelling.ocp import OptimalControlProblem

if __name__ == '__main__':
    from yaocptool.methods import DirectMethod


class LinearSystem(SystemModel):
    def __init__(self):
        SystemModel.__init__(self, Nx =2, Ny= 0, Nu=1)
        
        x_1 = self.x_sym[0]
        x_2 = self.x_sym[1]
        y = self.y_sym
        u = self.u_sym
        
        ode = vertcat(
                        x_2,
                        -x_1 -x_2 - u,
                      )
        alg = vertcat(((1-x_2**2)*x_1 - x_2 - y ))
        
        self.include_system_equations(ode, alg)


class Stabilizing(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1,1]), 'R':1 , 'x_ref':DM([0,0])}  
        self.state_constraints = False
        self.control_constraints = False        
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5
        self.x_0 = [0, 1]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[0] = 2
            self.x_max[1] = -0.4
            self.x_max[1] = 2
            self.x_min[1] = -0.4
        if self.control_constraints:
            self.u_max[0] = 1
            self.u_min[0] = -0.3


        
if __name__ == '__main__':
    nlp_prob, nlp_call = 0, 0 
    model = LinearSystem()
    problem = Stabilizing(model)
    
#    indir_method = IndirectMethod.IndirectMethod(problem, degree = 1, finite_elements = 50, integrator_type = 'implicit')
#    x_sol, u_sol, V_sol = indir_method.solve()
#    x, u, t= indir_method.plot_simulate(x_sol, u_sol, [{'x':[0],'u':[0]},{'x':[2]}], 5)
    
    dirm = DirectMethod(problem, degree = 5, degree_control = 5, finite_elements = 50, integrator_type = 'implicit')
    x_sol, u_sol, V_sol = dirm.solve()
    x, y, u, t= dirm.plot_simulate(x_sol, u_sol, [{'x':[0], 'u':[0]}, {'x':[2]}], 5)

#    grad = GradientMethod.GradientMethod(problem, degree = 1, max_iter = 10, finite_elements = 50, integrator_type = 'implicit')
#    x_sol, u_sol, V_sol = grad.solve()    
#    x, u, t= grad.plot_simulate(x_sol, u_sol, [{'x':[0],'u':[0]},{'x':[1]}], 5)
## it.:  9 cost:  0.682064

#    aug = AugmentedLagrange.AugmentedLagrangian(problem, IndirectMethod.IndirectMethod, \
#        { 'integrator_type': 'explicit'},
#            max_iter = 5, mu_0 = 1, finite_elements = 20, degree = 5)
#    x_sol, u_sol, V_sol =aug.solve()
    
#    x, u, t= aug.plot_simulate(x_sol, u_sol, [{'x':[2]},{'u':[1]}], 3, integrator_type = 'implicit')

#    print x_sol[1]
#    a= aug.model.simulate(x_sol[0], t_0 = 0.0,t_f = 5./aug.finite_elements, integrator_type= 'implicit')['xf']
#    print a


#    nlp_prob, nlp_call = aug.ocp_solver.multipleShootingScheme(p = aug.mu, theta = aug.nu)
#    ms = solutionmethods.nlp_prob
#    F = Function('f',[ms['x'],ms['p']],[ms['g']])
#    F(V_sol, vertcat(aug.mu, vec(aug.nu.values())))