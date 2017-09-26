# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")

from casadi import SX,DM, inf, repmat, vertcat, log, \
                    substitute, exp, jacobian, Function, gradient, depends_on, dot, rootfinder, hessian
import time
import AugmentedLagrange
class SaturationFunctionMethod(AugmentedLagrange.AugmentedLagrange):
    ''' For a minmization problem in the form
            min f(x,u) = \int L(x,u) dt
            s.t.: \dot{x} = f(x,u),
            g_ineq (x,u) \leq 0
        
        Transforms the problem in a sequence of solution of the problem
            min f(x,u) = \int L(x,u) -\mu \sum \log(-g_ineq(x,u)) dt
            s.t.: \dot{x} = f(x,u),
    '''
    
    def __init__(self, problem, r_eq_list, Ocp_solver_class, solver_options ={}, **kwargs):
        self.problem = problem

        self.Nr = 0        
        self.nu_sym = []
        self.mu_sym = SX.sym('mu')
        
        self.max_iter = 3
        self.degree = 3
        self.finite_elements = 20
        self.integrator_type = 'implicit'
        self.mu_0 = 10.
        self.nu = None      
        self.new_nu_funct = None
        self.beta = 4.
        self.mu_max = self.mu_0*self.beta**4

        self.tol = 1e-4
        self.m = 0

        self.relax_state_bounds = True
        self.relax_algebraic_equations = True 
#        self.relax_control_bounds = False
        self.parametrize = True
        
        self.solver = None

        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        self.mu = self.mu_0    
        
        self.includeSaturationFunction(r_eq_list)
        
        if self.model.Ny >0 and self.relax_algebraic_equations:
            self.relax_algebraic_equations()
        
        if self.relax_state_bounds:
            self.relax_states_constraints()


        self.model.includeParameter(self.mu_sym)  
        if self.parametrize:
            self.parametrize_nu()

        if self.nu == None:
            self.initialize_nu_values()
        
        # Initialize OCP solver        
        solver_options['finite_elements'] = self.finite_elements     
        solver_options['integrator_type'] = self.integrator_type
        
        self.ocp_solver = Ocp_solver_class(self.problem, **solver_options)        
        
    def includeSaturationFunction(self, r_eq_list):
        ''' r_eq_list is a tuple (n, eq) where r is the constraint order and 
            eq is the r-th derivative of the constraint
        '''
        i = 0
        k = 4#4
#        psi_max = SX.sym('psi_max')
#        psi_min = SX.sym('psi_min')
#        xi_foo = SX.sym('xi_foo')
        psi = lambda xi_foo, psi_max, psi_min: psi_max - (psi_max-psi_min)/(1+exp(k*xi_foo/(psi_max-psi_min)))
        
        for (r, eq, eq_max, eq_min) in r_eq_list:
            r = 0
            d_eq = [eq]
            while not depends_on(eq, self.model.u_sym):
                eq = dot(gradient(eq, self.model.x_sym), self.model.ode)
                d_eq.append(eq)
                r += 1
            d_eq = vertcat(*d_eq)
            
            xi = SX.sym('xi_'+ `i`, r)
            xi_0_sym = SX.sym('xi_0_sym_'+ `i`, r)
            v = SX.sym('v_'+`i`)
            
            if r>1:
                ode = vertcat(xi[1:r], v)
            else:
                ode = v
                
            h_list =[eq_max - (eq_max-eq_min)/(1+exp(4*xi[0]/(eq_max-eq_min)))]
#            h_list.append()            
            for j in range(r):
                h_list.append(dot(gradient(h_list[j],xi),ode))
#            h_list.append(gradient(h_list[0], xi[0])*xi[1])
#            for j in range(1, r):
#                gama = sum([gradient(h_list[j], xi[k])* xi[k+1] for k in range(j)])
#                if j == r-1:
#                    h_list.append(gama + gradient(h_list[0], xi[0])*v)
#                else:
#                    h_list.append(gama + gradient(h_list[0], xi[0])*xi[j+1])
            h_list = vertcat(*h_list)
                
            c_init = substitute(d_eq[:-1], self.model.x_sym, self.model.x_0_sym)
            h_init = h_list[:-1]
            initial_cond = c_init - h_init
            
#            xi_0 = rf_init_cond(0)
#            print xi_0
            self.problem.includeState(xi, ode, x_0_sym = xi_0_sym, h_initial = initial_cond)
            self.problem.includeAlgebraic(v, d_eq[-1]- h_list[-1])
#            self.problem.includeAlgebraic(v, eq- f)
            

            self.problem.L +=  v**2/self.mu_sym

            
        