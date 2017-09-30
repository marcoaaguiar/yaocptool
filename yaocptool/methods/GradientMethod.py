# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 14:03:59 2016

@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 16:39:52 2016

@author: marco
"""
import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.1.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import MX, DM, inf, vertcat, collocation_points, \
                    substitute, linspace, vec, gradient, hessian, mtimes, inv, fmin, fmax, Function
#from cartpendulum import *

from yaocptool.methods.classic import indirectmethod
import warnings


class GradientMethod(indirectmethod.IndirectMethod):
    def __init__(self, problem, **kwargs):
        indirectmethod.IndirectMethod.__init__(self, problem)

        self.solution_class = 'gradient_method'
        self.degree = 1
        self.max_iter = 1

        self.alpha_1 = 1e-2
        self.alpha_3 = 1e-1
        self.alpha = self.alpha_2
        self.eps_c = 1e-5
        self.eps_g = 1e-6        
        self.kappa_m = 2/3.
        self.kappa_p = 3/2.
        self.eps_alpha_m = 0.1
        self.eps_alpha_p = 0.9
        self.alpha_min = 1e-5
        self.alpha_max = 1.
        self.U = None
        self.parametrized_control = False
        self.hasCostState = False
        
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        self.checkBounds()
    
    @property
    def alpha_2(self):
        return (self.alpha_1+ self.alpha_3)/2
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
                
    def numberOfVariables(self):
        return self.model.Nx*(self.finite_elements+1) \
            + self.finite_elements*self.model.Nu*self.degree

        
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
            

    def initializeU(self):
        if self.U is None:
            self.U = []
            for i in range(self.finite_elements):
                self.U.append(vec(DM.zeros(self.model.Nu, self.degree)))
        
    def defineS(self):
        self.s = -gradient(self.problem.H, self.model.u_sym)
        
    def prepare(self):
        self.includeAdjointStates()
        self._create_cost_state()
        self.initializeU()
        self.defineS()
        self.parametrizeControl()
#        self.u_opt = self.calculateOptimalControl()
#        self.ReplaceWithOptimalControl(self.u_opt)
        

    def get_solver(self, num_parameters=None, initial_condition_as_parameter = False):
        ''' 
            all_mx = [p, theta, x_0]
        '''
        if num_parameters is not None:
            raise Exception('option deprecated')
            
        if not self.prepared:
            self.prepare()
            self.prepared = True
        
        if self.solver is None:
            if self.model.Np + self.model.Ntheta>0 or initial_condition_as_parameter:
                p_mx = MX.sym('p', self.model.Np)
                
                theta_mx = MX.sym('theta_',self.model.Ntheta, self.finite_elements)
                theta = dict([(i, vec(theta_mx[:,i])) for i in range(self.finite_elements)])
                
                all_mx =vertcat(p_mx, vec(theta_mx))
                if initial_condition_as_parameter:
                    p_mx_x_0 = MX.sym('x_0_p',self.model.Nx)
                    all_mx = vertcat(all_mx, p_mx_x_0)
                else:
                    p_mx_x_0 = None
                
                
                nlp_prob, nlp_call = self.multipleShootingScheme(p=p_mx, x_0=p_mx_x_0, theta = theta)
                
                nlp_prob['p'] = all_mx
    
            else:
                nlp_prob, nlp_call = self.multipleShootingScheme()
            
            self.nlp_prob = nlp_prob
            self.nlp_call = nlp_call
            self.solver = self.createNumSolver(nlp_prob)
            
        return self.callSolver
    def getForwardBackward(self, direction):
        dae = self.model.getDAESystem()
        if direction == 'f':
#            dae['x'] = vertcat(dae['x'][:(self.model.Nx-1)/2],dae['x'][-1])
#            dae['ode'] = vertcat(dae['ode'][:(self.model.Nx-1)/2],dae['ode'][-1])
            pass
        if direction == 'b':
#            dae['x'] = dae['x'][self.model.Nx/2:]
            dae['ode'] = -dae['ode']
        return dae
            
    def forwardSimulation(self, U):
#        t_list = (linspace(self.problem.t_0, self.problem.t_f, self.finite_elements+1))
#        col_points = collocation_points(self.degree, 'radau')
        h = (self.problem.t_f - self.problem.t_0)/self.finite_elements
#        delta_t_list = [col_points[j]*h for j in range(self.degree)]
        
#        x_0 = vertcat(self.problem.x_0[:(self.model.Nx-1)/2],self.problem.x_0[-1])
        x_0 = self.problem.x_0
        X = [x_0]
        
            
        for k in range(self.finite_elements):
            dae = self.getForwardBackward('f')
#            dae['x'] = vertcat(dae['x'], x_c)
#            dae['ode'] = vertcat(dae['ode'], ode_x_c)
            t_0 = k*h
            t_f = (k+1)*h
            self.model.convertFromTauToTime(dae, t_0, t_f)
            XF = self.model.simulateStep(X[k], t_0 = t_0, t_f = t_f, p = U[k], 
                                         dae_sys = dae,
#                                         dae_sys = None,
                                         integrator_type = self.integrator_type)
            X.append(XF)
        return X

    def backwardSimulation(self, U, X):
        t_list = (linspace(self.problem.t_0, self.problem.t_f, self.finite_elements+1))
        col_points = collocation_points(self.degree, 'radau')
        h = (self.problem.t_f - self.problem.t_0)/self.finite_elements
        delta_t_list = [col_points[j]*h for j in range(self.degree)]
        
        f = Function('lamb_0', [self.model.x_sym], [-self.problem.h_final])
        
        lamb_0 = f(vertcat(X[-1][:(self.model.Nx-1)/2],DM.zeros((self.model.Nx-1)/2), X[-1][-1]))
        x_0 = vertcat(X[-1][:(self.model.Nx-1)/2], lamb_0, X[-1][-1])      
        Lamb = [x_0]
        
        UB= list(reversed(U))
        XB = list(reversed(X))
        for k in range(self.finite_elements):
            dae = self.getForwardBackward('b')
            t_0 = k*h
            t_f = (k+1)*h
            self.model.convertFromTauToTime(dae, t_k= t_0, t_kp1 = t_f)
            XF = self.model.simulateStep(x_0, t_0 = t_0, t_f = t_f, p = UB[k], 
                                         dae_sys = dae,
#                                         dae_sys = None,
                                         integrator_type = self.integrator_type)
                                         
            Lamb.append(XF)
#            x_0 = vertcat(XB[k][:4], XF[4:-1], XB[k][-1])
            x_0 = XF
#            for j in range(self.degree):                
#                I = self.model.createIntegrator(dae, {'t0': float(t_list[k]),
#                       'tf': float(t_list[k]+ delta_t_list[j])}, integrator_type = self.integrator_type)
#                       
#                u = U[self.finite_elements-1 -k]
#                x_f = I(x0 = x_0, p = u)['xf']
        Lamb = list(reversed(Lamb))
        return Lamb 
    def costFunctionSimulation(self, U):
        cost = self.forwardSimulation(U)[-1][-1]
        return cost
    def computeSearchDirection(self, X_Lamb):
        t_list = (linspace(self.problem.t_0, self.problem.t_f, self.finite_elements+1))
        col_points = collocation_points(self.degree, 'radau')
        h = (self.problem.t_f - self.problem.t_0)/self.finite_elements
        delta_t_list = [col_points[j]*h for j in range(self.degree)]
        
        S = {}
        for i in range(self.finite_elements):
            S[i] = DM.zeros(self.model.Nu, self.degree)
        
        for k in range(self.finite_elements):
            s = self.model.convertExprFromTauToTime(self.s, t_list[k], t_list[k+1])
            S_Func = Function('s_func', [self.model.x_sym, self.model.t_sym, 
                                            self.model.p_sym, self.model.theta_sym,
                                            self.model.u_par], [s])
            for j in range(self.degree):
                x = X_Lamb[k*self.degree +j]
                bla = S_Func(x, float(t_list[k]+ delta_t_list[j]), [], [], self.U[k])
                S[k][:,j] = bla
            S[k] = vec(S[k])
        return S
    def computeStepSize(self, U, S):
#        alpha_sym = MX.sym('alpha_sym')
#        solver = nlpsol('alpha_solver', 'ipopt', {'f':self.computeCostValue(U, S, alpha_sym), 'x':alpha_sym})
#        self.alpha = solver(x0 = self.alpha)['x']
#        return self.alpha

        J_1 = self.computeCostValue(U, S, self.alpha_1)
        J_2 = self.computeCostValue(U, S, self.alpha_2)
        J_3 = self.computeCostValue(U, S, self.alpha_3)
        
#        print self.alpha_1, self.alpha_2, self.alpha_3
        print J_1, J_2, J_3
        
#        if not J_1.is_nan() or not J_2.is_regular() or not J_3.is_regular():
#            raise Exception(U, S)
        c_0, c_1, c_2 = self.polynomialCoefficients( J_1, J_2, J_3)
#        if c_2 >0:
#            raise Exception('Not convex')
        
        alpha = self.approximateStepSize(c_0, c_1, c_2, J_1, J_2, J_3)
        self.updateAlpha(alpha)
        return alpha
        
    def polynomialCoefficients(self, J_1, J_2, J_3):
        alpha_1 = self.alpha_1
        alpha_2 = self.alpha_2
        alpha_3 = self.alpha_3
        
        c_0 = (alpha_1*(alpha_1 - alpha_2)*alpha_2*J_3 + alpha_2*alpha_3*(alpha_2 - alpha_3)*J_1 + alpha_1*alpha_3*(alpha_3 - alpha_1)*J_2)/((alpha_1 - alpha_2)*(alpha_1 - alpha_3)*(alpha_2 - alpha_3))
        c_1 = ((alpha_2**2 - alpha_1**2)*J_3 + (alpha_1**2 - alpha_3**2)*J_2 + (alpha_3**2 - alpha_2**2)*J_1)/((alpha_1 - alpha_2)*(alpha_1 - alpha_3)*(alpha_2 - alpha_3))
        c_2 = ((alpha_1 - alpha_2)*J_3 + (alpha_2 - alpha_3)*J_1 + (alpha_3 - alpha_1)*J_2)/((alpha_1 - alpha_2)*(alpha_1 - alpha_3)*(alpha_2 - alpha_3))
        
        return c_0, c_1, c_2
    def approximateStepSize(self, c_0, c_1, c_2, J_1, J_2, J_3):
        alpha_hat =  - c_1/(2.*c_2)
        if c_2 > self.eps_c:
#        a = alpha_hat
            alpha_hat = fmin(self.alpha_3, fmax(self.alpha_1, alpha_hat))
#        alpha_hat = fmax(alpha_hat, self.alpha_min)
#        print a, alpha_hat
        else:
            if J_1 + self.eps_g< fmin(J_2, J_3):
                alpha_hat = self.alpha_1
            elif J_3 + self.eps_g < fmin(J_1, J_2):
                alpha_hat = self.alpha_3
            else:
                alpha_hat = self.alpha_2 
                
        return alpha_hat

    def updateAlpha(self, alpha):
        interval = DM([self.alpha_1, self.alpha_3])
        if alpha >= self.alpha_1 + self.eps_alpha_p*(self.alpha_3 - self.alpha_1) and self.alpha_3 <= self.alpha_max:
            interval = self.kappa_p*interval
        elif alpha <= self.alpha_1 + self.eps_alpha_m*(self.alpha_3 - self.alpha_1) and self.alpha_1 >= self.alpha_min:
            interval = self.kappa_m*interval
        else:
            pass
        self.alpha_1 = interval[0]
        self.alpha_3 = interval[1]
        
    def applyPsiToU(self, U):
        new_U = []
        for k in range(len(U)):
            new_U.append(fmax(self.problem.u_min, fmin(self.problem.u_max, U[k])))
        return new_U
    def applyStoU(self, U, S, alpha = 1):
        new_U = []
        for k in range(len(U)):
            new_U.append(U[k] + S[k]*alpha)
        return new_U
        
    def computeCostValue(self, U, S, alpha):
        applied_control = self.applyStoU(U, S, alpha)
        applied_control = self.applyPsiToU(applied_control)
        cost = self.costFunctionSimulation(applied_control)
        return cost
    def getNewControl(self, U, S, alpha):
        new_control = self.applyStoU(U, S, alpha)
        new_control = self.applyPsiToU(new_control)         
        return new_control
    def solve(self):
        if not self.prepared:
            self.prepare()
            
        X = self.forwardSimulation(self.U)
        X_Lamb = X
        S =[]
        for i in range(self.max_iter):
            X_Lamb = self.backwardSimulation(self.U, X)
            print X_Lamb[0][:4], X[0][:4]
            S = self.computeSearchDirection(X_Lamb)
            Alpha = self.computeStepSize(self.U, S)
            self.U = self.getNewControl(self.U, S, Alpha)
            chosen = self.computeCostValue(self.U, S, Alpha)
            print 'it.: ', i, 'cost: ', chosen
            X = self.forwardSimulation(self.U)
        return X_Lamb, self.U, S     
if __name__ == '__main__':
    model2 = PendulumCart() 
    
#    problem = DownwardPendulumStabilization(model)
    problem2 = UpwardPendulumStabilization(model2)
    problem2.t_f = 1.

    problem2.V = 0.
    problem2.u_max[0] = 20
    problem2.u_min[0] = -20
#    problem.x_0 = [pi+0.2, 0, 0,0]
#    problem.x_0 = [0.0002, 0, 0,0]
    
#    grad = GradientMethod(problem2, degree = 1, max_iter = 1, U = U, finite_elements = 40, integrator_type = 'implicit')
    grad = GradientMethod(problem2, degree = 1, max_iter =5, finite_elements = 40, integrator_type = 'implicit')
    x_sol, u_sol, V_sol = grad.solve()
    x, u, t= grad.plotSimulate(x_sol, u_sol, [{'x':[0]},{'x':[1,2,3]},{'u':[0]}], 10)
    
