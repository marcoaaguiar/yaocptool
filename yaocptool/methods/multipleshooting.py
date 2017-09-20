# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from casadi import DM, MX, vertcat, Function, repmat
from discretizationschemebase import DiscretizationSchemeBase


class MultipleShootingScheme(DiscretizationSchemeBase):
    def number_of_variables(self):
        return self.model.Nx * (self.finite_elements + 1) \
               + self.finite_elements * self.model.Nu * self.degree_control \
               + self.problem.N_eta

    def create_nlp_symbolic_variables_and_bound_vectors(self):
        NV = self.number_of_variables()

        V = MX.sym("V", NV)
        vars_lb = -DM.inf(NV)
        vars_ub = DM.inf(NV)

        X, U = self.splitXandU(V)

        v_offset = 0
        for k in range(self.finite_elements + 1):
            vars_lb[v_offset:v_offset + self.model.Nx] = self.problem.x_min
            vars_ub[v_offset:v_offset + self.model.Nx] = self.problem.x_max
            v_offset = v_offset + self.model.Nx

            if k != self.finite_elements:
                for j in range(self.degree_control):
                    vars_lb[v_offset:v_offset + self.model.Nu] = self.problem.u_min
                    vars_ub[v_offset:v_offset + self.model.Nu] = self.problem.u_max
                    v_offset = v_offset + self.model.Nu
        eta = V[NV - self.problem.N_eta:]
        return V, X, U, eta, vars_lb, vars_ub

    def splitXandU(self, V):
        X = []
        U = []
        v_offset = 0
        if self.problem.N_eta > 0:
            V = V[:-self.problem.N_eta]

        for k in range(self.finite_elements + 1):
            X.append(V[v_offset:v_offset + self.model.Nx])
            v_offset = v_offset + self.model.Nx
            if k != self.finite_elements:
                U.append(V[v_offset:v_offset + self.model.Nu * self.degree_control])
                v_offset = v_offset + self.model.Nu * self.degree_control
        return X, U

    def discretize(self, finite_elements=None, x_0=None, p=[], theta=None):
        finite_elements = self.finite_elements

        if theta is None:
            theta = dict([(i, []) for i in range(finite_elements)])

        if x_0 == None:
            x_0 = self.problem.x_0

        t0 = self.problem.t_0
        tf = self.problem.t_f
        h = (tf - t0) / finite_elements

        # Get the state at each shooting node
        V, X, U, eta, vars_lb, vars_ub = self.create_nlp_symbolic_variables_and_bound_vectors()
        G = []

        F_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        F_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])

        G.append(F_h_initial(X[0], x_0))

        for k in range(finite_elements):
            iopts = {}
            iopts["t0"] = k * h
            iopts["tf"] = (k + 1) * h

            dae_sys = self.model.getDAESystem()
            self.model.convertFromTauToTime(dae_sys, k * h, (k + 1) * h)

            p_i = vertcat(p, theta[k], U[k])

            #            I = self.model.createIntegrator(dae_sys, iopts, integrator_type= self.integrator_type)
            #            XF = I(x0=X[k], p = p_i)["xf"]
            XF = self.model.simulateStep(X[k], t_0=k * h, t_f=(k + 1) * h, p=p_i, dae_sys=dae_sys,
                                         integrator_type=self.solution_method.integrator_type)

            G.append(XF - X[k + 1])

        G.append(F_h_final(X[-1], eta))

        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(X[-1], p)
        else:
            cost = 0
        nlp_prob = {}
        nlp_call = {}

        nlp_prob['g'] = vertcat(*G)
        nlp_prob['x'] = V
        nlp_prob['f'] = cost
        nlp_call['lbx'] = vars_lb
        nlp_call['ubx'] = vars_ub
        nlp_call['lbg'] = DM.zeros(nlp_prob['g'].shape)
        nlp_call['ubg'] = DM.zeros(nlp_prob['g'].shape)

        return nlp_prob, nlp_call

    def create_initial_guess(self):
        base_x0 = self.problem.x_0
        base_x0 = vertcat(base_x0, repmat(DM([0] * self.model.Nu), self.degree_control))
        x0 = vertcat(repmat(base_x0, self.finite_elements), self.problem.x_0)
        x0 = vertcat(x0, DM.zeros(self.problem.N_eta))
        return x0
