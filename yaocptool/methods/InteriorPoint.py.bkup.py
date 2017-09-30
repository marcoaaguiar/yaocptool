# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 16:53:36 2016

@author: marco
"""

import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")

from casadi import SX,DM, inf, repmat, vertcat, log, \
                    sum1, fmax
import time
class InteriorPoint:
    ''' For a minmization problem in the form
            min f(x,u) = \int L(x,u) dt
            s.t.: \dot{x} = f(x,u),
            g_ineq (x,u) \leq 0
        
        Transforms the problem in a sequence of solution of the problem
            min f(x,u) = \int L(x,u) -\mu \sum \log(-g_ineq(x,u)) dt
            s.t.: \dot{x} = f(x,u),
    '''
    
    def __init__(self, problem, Ocp_solver_class, solver_options ={}, **kwargs):
        self.problem = problem
        
        self.max_iter = 3
        self.mu_0 = 10.
        self.beta = 4.
        self.tol = 1e-4
        self.m = 0

        self.relax_state_bounds = True
        self.relax_algebraic_bounds = True 
        self.relax_control_bounds = False
        self.solver = None
        
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        self.mu = self.mu_0        
        self.mu_min = self.mu_0/self.beta**4
        self.includeLogBarrier()
        
        self.ocp_solver = Ocp_solver_class(self.problem, **solver_options)
    @property
    def model(self):
        return self.problem.model
    @property
    def splitXandU(self):
        return self.ocp_solver.splitXandU
    @property
    def joinXandU(self):
        return self.ocp_solver.joinXandU
        
    def includeLogBarrier(self):
        mu_sym = SX.sym('mu')
        self.mu_sym = mu_sym
        self.model.includeParameter(mu_sym)        
        
        self.problem.L += -mu_sym*sum1(log(-self.problem.g_ineq))
        
            # For state variables
        if self.relax_state_bounds:
            for i in range(self.model.Nx):        
                if self.problem.x_max[i] != inf:
#                    self.problem.L += -mu_sym*log(-self.model.x_sym[i] + self.problem.x_max[i])
                    x_log = SX.sym('x_log',1)
                    self.problem.includeState(x_log, -mu_sym*log(-self.model.x_sym[i] + self.problem.x_max[i]), 0)
                    self.problem.V += x_log
                    self.problem.x_max[i] = inf
                    self.m +=1

                if self.problem.x_min[i] != -inf:
#                    self.problem.L += -mu_sym*log(+self.model.x_sym[i] - self.problem.x_min[i])
                    x_log = SX.sym('x_log',1)
                    self.problem.includeState(x_log, -mu_sym*log(+self.model.x_sym[i] - self.problem.x_min[i]), 0)
                    self.problem.V += x_log
                    self.problem.x_min[i] = -inf
                    self.m +=1

            #For algebraic variable
        if self.relax_algebraic_bounds:
            for i in range(self.model.Ny):
                if self.problem.y_max[i] != inf:
                    self.problem.L -= mu_sym*log(-self.model.y_sym[i] + self.problem.y_max[i])
                    self.problem.y_max[i] = inf
                    self.m +=1
                    
                if self.problem.y_min[i] != -inf:
                    self.problem.L -= mu_sym*log(+self.model.y_sym[i] - self.problem.y_min[i])
                    self.problem.y_min[i] = -inf
                    self.m +=1
                
            #For control variables
        if self.relax_control_bounds:
            for i in range(self.model.Nu):
                if self.problem.u_max[i] != inf:
                    self.problem.L -= mu_sym*log(-self.model.u_sym[i] + self.problem.u_max[i])
                    self.problem.u_max[i] = inf
                    self.m +=1

                if self.problem.u_min[i] != -inf:
                    self.problem.L -= mu_sym*log(+self.model.u_sym[i] - self.problem.u_min[i])
                    self.problem.u_min[i] = -inf
                    self.m +=1

        self.problem._g_ineq = self.problem.g_ineq
        self.problem.g_ineq = []
    
    def simulate(self, X, U, sub_elements = 5, t_0 = None, t_f =None, p = [], theta = None, integrator_type = 'implicit'): 
        if t_0 is None:
            t_0 = self.problem.t_0
        if t_f is None:
            t_f = self.problem.t_f
            
        par = vertcat(p, self.mu)
        
        micro_X, micro_U, micro_t =  self.ocp_solver.simulate(X, U, sub_elements, t_0, t_f, par, theta, integrator_type = integrator_type)
        return micro_X, micro_U, micro_t
        
    def solve(self, initial_guess = None,  p=[], theta = None, x_0 = []):
        V_sol = self.solve_raw(initial_guess)
        return self.ocp_solver.splitXandU(V_sol)
        
    def solve_raw(self, initial_guess = None,  p=[], theta = None, x_0 = []):
        t1 = time.time()
        V_sol = initial_guess
        if self.solver is None:
            if len(DM(x_0).full())>0:
                initial_condition_as_parameter = True
            else: 
                initial_condition_as_parameter = False
            self.getOCPSolver(initial_condition_as_parameter)
            
        self.mu = self.mu*self.beta
        for i in range(self.max_iter):
            self.mu = fmax(self.mu_min, self.mu/self.beta)
#            V_sol = self.ocp_solver.solve_raw(x0 = V_sol, p=[self.mu])
            V_sol = self.solver(V_sol, p = self.mu, x_0 = x_0) 
            if self.m*self.mu <= self.tol:
                print 'break'
                break
        print 'Solution: ', time.time()-t1
        return V_sol
        
    def getSolver(self, initial_condition_as_parameter = False):
        self.getOCPSolver(initial_condition_as_parameter)
        return self.solve_raw
    
    def getOCPSolver(self,initial_condition_as_parameter = False):
        self.solver = self.ocp_solver.get_solver(initial_condition_as_parameter= initial_condition_as_parameter)
        
        
    def stepForward(self):
        pass
        
        
        
        
        
        