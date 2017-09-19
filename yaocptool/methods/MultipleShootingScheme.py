# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from casadi import DM, vertcat, Function
class MultipleShootingScheme:
    def discretize(self, finite_elements = None, x_0 = None, p = [], theta =None):
        if finite_elements == None:
            finite_elements = self.finite_elements 
        else:
            self.finite_elements = finite_elements
        
        if theta == None:
            theta = dict([(i,[]) for i in range(finite_elements)])
        
        if x_0 == None:
            x_0 = self.problem.x_0 
            
        t0 = self.problem.t_0
        tf = self.problem.t_f
        h = (tf-t0)/finite_elements
        
        # Get the state at each shooting node
        V, X,U, eta, vars_lb, vars_ub  = self.initializeNLPVariables()
        G = []
        
        F_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        F_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])
        
        G.append(F_h_initial(X[0], x_0))
        
        for k in range(finite_elements):
            iopts = {}
            iopts["t0"] = k*h
            iopts["tf"] = (k+1)*h
    
            dae_sys = self.model.getDAESystem()
            self.model.convertFromTauToTime(dae_sys, k*h, (k+1)*h)            

            p_i = vertcat(p, theta[k], U[k])

#            I = self.model.createIntegrator(dae_sys, iopts, integrator_type= self.integrator_type)
#            XF = I(x0=X[k], p = p_i)["xf"]
            XF = self.model.simulateStep(X[k], t_0 = k*h, t_f = (k+1)*h, p = p_i, dae_sys = dae_sys, integrator_type = self.integrator_type)
            
            G.append(XF-X[k+1])
        
        G.append(F_h_final(X[-1], eta))        
        
        if self.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym],[self.problem.V])(XF, p) 
        else: cost = 0
        nlp_prob = {}
        nlp_call = {}
        
        nlp_prob['g'] = vertcat(*G)
        nlp_prob['x'] = V
        nlp_prob['f'] = cost
        nlp_call['lbx']= vars_lb
        nlp_call['ubx']= vars_ub
        nlp_call['lbg'] = DM.zeros(nlp_prob['g'].shape)
        nlp_call['ubg'] = DM.zeros(nlp_prob['g'].shape)
        
        return nlp_prob, nlp_call