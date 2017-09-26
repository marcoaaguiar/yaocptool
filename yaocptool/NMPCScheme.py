# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 12:41:22 2016

@author: marco
"""

from problems.cartpendulum import *
from methods import DirectMethod, IndirectMethod, AugmentedLagrange
import matplotlib.pyplot as plt
from casadi import inf
import time

class NMPCScheme:
    def __init__(self, plant, problem, ocp_solver, x_0 = None, **kwargs):
        self.plant = plant
        self.problem = problem
#        self.ocp_solver = solutionmethods.SolutionMethodsBase()
        self.ocp_solver = ocp_solver
        
        
        self.t_0 = 0.
        self.t_f = 1.
        
        self.dt = 0.1
        
        self.max_steps = inf       
        self.verbose = 1  
        self.times = []
        
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        if x_0 ==None:
            self.x_0 = self.problem.x_0
        else:
            self.problem.x_0 = x_0
    
    def nextInitialGuess(self, V, x_0):
        x, u = self.ocp_solver.splitXandU(V)
        new_x = [x_0]
        new_x.extend(x[2:])
        new_x.extend(x[-1:])
        
        new_u = u[1:]
        new_u.extend(u[-1:])
        
        new_V = self.ocp_solver.joinXandU(new_x, new_u)
        return new_V
    def getControls(self, X, U, t_0, t_f, p = [], theta = None, sub_elements = 1):
        x, u, t = self.ocp_solver.simulate(X[:2], U[:1], sub_elements, t_0, t_f, p, theta)
        return u[0]
    def plot(self, X, U, plot_list, t_states):
        for entry in plot_list:
            if 'x' in entry:
                for i in entry['x']:
                    plt.step(t_states, horzcat(*X)[i,:].T, where='post')
            if 'u' in entry:
                for i in entry['u']:
                    plt.step(t_states[:len(U)], horzcat(*U)[i,:].T, where='post')
            plt.grid()
            plt.show()
                
    def run(self):
        t = self.t_0
        k = 0
        V_sol = 0
        solver = self.ocp_solver.get_solver(initial_condition_as_parameter=True)
        x_0 = self.problem.x_0
        X = [x_0]
        U = []
        T = [t]
        while t<self.t_f and k < self.max_steps:
            if self.verbose>=2:
                print 'Starting iteration: ',k

            tic = time.time()
            V_sol = solver(V_sol, x_0 = x_0)
            if self.verbose>=3:
                print 'optimizing'
            x,u = self.ocp_solver.splitXandU(V_sol)
            control = self.getControls(x,u, t_0= t, t_f = t+self.dt)[:self.plant.Nu]
#            return control, None, None
            #simulate
            if self.verbose>=3:
                print 'simulating'
            x_f_sim = self.plant.simulate(x_0 = x_0[:self.plant.Nx], t_f = t+self.dt, t_0=t, p = control, integrator_type = 'implicit')
            if self.verbose>=3:
                print 'simulated'
            x_0 = x[1]
            x_0[:self.plant.Nx] = x_f_sim
#            x_0[-1] = 0
            V_sol = self.nextInitialGuess(V_sol, x_0)
            X.append(x_0)
            U.append(control)
            T.append(t + self.dt)
#            x_0 = x[1]
            
            t = t + self.dt
            k +=1
            if self.verbose>=3:
                print 'next step'
            self.ocp_solver.stepForward()
            self.times.append(time.time()-tic)

        if self.verbose>=1:       
            print 'Total solution time: ', sum(self.times)
            print 'First it. solution time: ', self.times[0]
            print 'Average solution time: ', sum(self.times[1:])/(len(self.times)-1)
        return X, U, T
            

if __name__ == '__main__':      
    plant = PendulumCart() 
    model = PendulumCart() 
#    problem = UpwardPendulumStabilization(model)
    problem = UpwardPendulumStabilization(model, state_constraints = True)
#    ocp_solver = IndirectMethod(problem, degree = 1, finite_elements = 40, integrator_type = 'explicit')
    
#    ocp_solver = DirectMethod(problem, degree = 1, finite_elements = 40, integrator_type = 'explicit')
#    
    ocp_solver = AugmentedLagrange(problem, IndirectMethod, \
        {'degree': 1,},
            max_iter = 1, mu_0 = 10, beta= 10., finite_elements = 40, degree = 5, integrator_type = 'explicit')    
#
    dt = (problem.t_f - problem.t_0)/ocp_solver.finite_elements
    nmpc = NMPCScheme(plant, problem, ocp_solver, t_f = 10., dt = dt)
    X, U, T = nmpc.run()
#    
    nmpc.plot(X, U, [{'x':[0]},{'x':[2]},{'u':[0]}], T)
## Indirect Method
#Total solution time:  5.41700005531
#First it. solution time:  0.815999984741
#Average solution time:  0.0677125006914
    
## Direct Method
#Total solution time:  5.88800001144
#First it. solution time:  0.135999917984
#Average solution time:  0.0736000001431
    
## Augm Lag 
#Total solution time:  5.80099987984
#First it. solution time:  0.469000101089
#Average solution time:  0.072512498498


### Direct
#Total solution time:  7.20399999619
#First it. solution time:  0.271999835968
#Average solution time:  0.0900499999523

### 3.1.0 - Aug
#Solution time:  0.0870001316071
#Total solution time:  9.06700015068
#First it. solution time:  0.413000106812
#Average solution time:  0.113337501884
