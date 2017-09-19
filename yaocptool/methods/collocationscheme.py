# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from casadi import DM, MX, repmat, vertcat, Function, jacobian, \
    vec, collocation_points, horzcat, substitute


class CollocationScheme():
    def getMethods(self):

        def numberOfVariables(self):
            return self.model.Nx * (self.finite_elements) * (self.degree + 1) \
                   + self.model.Nyz * self.finite_elements * self.degree \
                   + self.model.Nu * self.finite_elements * self.degree_control \
                   + self.problem.N_eta

        def initializeNLPVariables(self):
            eta = MX.sym('eta', self.problem.N_eta)
            X = []
            Y = []
            U = []
            vars_lb = []
            vars_ub = []
            for k in range(self.finite_elements):
                X_k = []
                for n in range(self.degree + 1):
                    X_k.append(MX.sym('x_' + `k` + '_' + `n`, self.model.Nx))
                    vars_lb.append(self.problem.x_min)
                    vars_ub.append(self.problem.x_max)
                X.append(X_k)
            for k in range(self.finite_elements):
                Y_k = []
                for n in range(self.degree):
                    Y_k.append(MX.sym('yz_' + `k` + '_' + `n`, self.model.Nyz))
                    vars_lb.append(self.problem.yz_min)
                    vars_ub.append(self.problem.yz_max)
                Y.append(Y_k)
            for k in range(self.finite_elements):
                U_k = []
                for n in range(self.degree_control):
                    U_k.append(MX.sym('u_' + `k` + '_' + `n`, self.model.Nu))
                    vars_lb.append(self.problem.u_min)
                    vars_ub.append(self.problem.u_max)
                U.append(vertcat(*U_k))

            V_x = vertcat(*[vertcat(*x_k) for x_k in X])
            V_y = vertcat(*[vertcat(*yz_k) for yz_k in Y])
            V_u = vertcat(*U)
            V = vertcat(V_x, V_y, V_u, eta)

            vars_lb = vertcat(*vars_lb)
            vars_ub = vertcat(*vars_ub)

            return V, X, Y, U, eta, vars_lb, vars_ub

        def splitXYandU(self, V, all_subinterval = False):
            '''
            :param V:  solution of NLP
            :param all_subinterval = False: Returns all elements of the subinterval (or only the first one)
            :return: X, Y, and U -> list with a DM for each element
            '''
            X = []
            Y = []
            U = []
            v_offset = 0
            if self.problem.N_eta > 0:
                V = V[:-self.problem.N_eta]

            for k in xrange(self.finite_elements):
                X_k = []
                for i in xrange(self.degree + 1):
                    X_k.append(V[v_offset:v_offset + self.model.Nx])
                    v_offset += self.model.Nx
                if all_subinterval:
                    X.append(X_k)
                else:
                    X.append(X_k[0])
            X.append(X_k[-1])
            for k in xrange(self.finite_elements):
                Y_k = []
                for i in xrange(self.degree):
                    Y_k.append(V[v_offset:v_offset + self.model.Nyz])
                    v_offset += self.model.Nyz
                if all_subinterval:
                    Y.append(Y_k)
                else:
                    Y.append(Y_k[0])


            for k in xrange(self.finite_elements):
                U_k = V[v_offset:v_offset + self.model.Nu*self.degree_control]
                v_offset += self.model.Nu*self.degree_control
                U.append(U_k)
            assert v_offset == V.numel()

            return X, Y, U

        def splitXandU(self, V, all_subinterval = False):
            '''
            :param V:  solution of NLP
            :param all_subinterval = False: Returns all elements of the subinterval (or only the first one)
            :return:
            '''
            X, Y, U = self.splitXYandU(V)
            return X, U

        def discretize(self, finite_elements=None, degree=None, x_0=None, p=[], theta=None):
            if finite_elements == None:
                finite_elements = self.finite_elements
            if degree == None:
                degree = self.degree
            if theta == None:
                theta = dict([(i, []) for i in range(finite_elements)])
            if x_0 == None:
                x_0 = self.problem.x_0

            t0 = self.problem.t_0
            tf = self.problem.t_f
            h = (tf - t0) / finite_elements

            V, X, YZ, U, eta, vars_lb, vars_ub = self.initializeNLPVariables()

            G = []

            F_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
            F_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])

            ###
            u_pol, u_par = self.createVariablePolynomialApproximation(self.model.Nu, self.degree_control, name='u_col',
                                                                      point_at_t0=False)
            # self.model.control_function = u_pol
            # self.model.u_par = u_par
            # self.u_pol = u_pol

            tau, L_list = self.createLagrangianPolynomialBasis(self.degree, starting_index=0)
            dL_list = [jacobian(l, tau) for l in L_list]
            f_dL_list = Function('f_dL_list', [tau], dL_list)

            G.append(F_h_initial(X[0][0], x_0))
            for n_element in range(self.finite_elements):
                dt = (self.problem.t_f - self.problem.t_0) / self.finite_elements
                t_0_element = dt * n_element
                t_f_element = dt * (n_element + 1)

                tau_list = [0] + collocation_points(self.degree)
                mapping = lambda tau: (t_0_element + tau * dt)
                micro_t_k = map(mapping, tau_list)

                # self.model.convertFromTauToTime(dae_sys, t_0_element, t_f_element)


                # f_x_pol = Function('f_x_pol', [self.model.tau_sym, x_par], [x_pol])
                # f_dx_pol = Function('f_dx_pol', [self.model.tau_sym, x_par], [dx_pol])

                # p_element = vertcat(p, theta[n_element], vec(U[n_element]))
                #                G.append(U[n_element] - DM([1,2]))
                if n_element != 0:
                    G.append(X[n_element - 1][-1] - X[n_element][0])

                for c_point in xrange(self.degree):
                    dae_sys = self.model.getDAESystem()
                    if not 'z' in dae_sys:
                        dae_sys['z'] = vertcat([])
                        dae_sys['alg'] = vertcat([])
                    if not 'p' in dae_sys:
                        dae_sys['p'] = vertcat([])

                    # dae_sys['ode']  = substitute(dae_sys['ode'], dae_sys['x'], x_pol)
                    # dae_sys['alg']  = substitute(dae_sys['alg'], dae_sys['x'], x_pol)

                    # dae_sys['ode']  = substitute(dae_sys['ode'], self.model.tau_sym, tau_list[i+1])
                    # dae_sys['alg']  = substitute(dae_sys['alg'], self.model.tau_sym, tau_list[i+1])


                    arg_sym = [self.model.tau_sym, dae_sys['x'], dae_sys['z'], dae_sys['p']]

                    f_ode = Function('f_ode', arg_sym, [dae_sys['ode']])
                    f_alg = Function('f_alg', arg_sym, [dae_sys['alg']])

                    p_i = vertcat(p, theta[n_element], U[n_element])

                    arg = [ tau_list[c_point + 1], X[n_element][c_point + 1], YZ[n_element][c_point], p_i ]
                    # f_x_arg = [micro_t_k[i + 1], vec(horzcat(*X[n_element]))]

                    d_x_d_tau = sum([f_dL_list(tau_list[c_point + 1])[k] * X[n_element][k] for k in range(degree + 1)])

                    G.append(d_x_d_tau - dt * f_ode(*arg))
                    G.append(f_alg(*arg))
                    arg
                # XF = f_x_pol(*f_x_arg)
                XF = X[n_element][-1]

            G.append(F_h_final(XF, eta))
            for item in G:
                print item
            if self.solution_class == 'direct':
                cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(XF, p)
            else:
                cost = 0

            nlp_prob = {}
            nlp_call = {}
            nlp_prob['g'] = vertcat(*G)
            nlp_prob['x'] = V
            #            nlp_prob['f'] = cost
            nlp_prob['f'] = cost
            nlp_call['lbx'] = vars_lb
            nlp_call['ubx'] = vars_ub
            nlp_call['lbg'] = DM.zeros(nlp_prob['g'].shape)
            nlp_call['ubg'] = DM.zeros(nlp_prob['g'].shape)

            return nlp_prob, nlp_call

        def createInitialGuess(self):
            base_x0 = repmat(self.problem.x_0, (self.degree + 1) * self.finite_elements)
            base_x0 = vertcat(base_x0,
                              DM.zeros((
                                       self.model.Nyz * self.degree * self.finite_elements + self.model.Nu * self.degree_control * self.finite_elements),
                                       1))

            base_x0 = vertcat(base_x0, DM.zeros(self.problem.N_eta))
            return base_x0

        def solve(self, initial_guess=None, p=[]):
            V_sol = self.solve_raw(initial_guess, p)

            X, U = self.splitXandU(V_sol)
            X_finite_element = []
            U_finite_element = []

            for x in X:
                X_finite_element.append(x[0])
            X_finite_element.append(X[-1][-1])
            #            for u in U:
            #                U_finite_element.append(u[0])

            return X_finite_element, U, V_sol

        methods = {'discretize': discretize,
                   'initializeNLPVariables': initializeNLPVariables,
                   'splitXandU': splitXandU,
                   'splitXYandU': splitXYandU,
                   'numberOfVariables': numberOfVariables,
                   'createInitialGuess': createInitialGuess,
                   'solve': solve
                   }
        return methods
