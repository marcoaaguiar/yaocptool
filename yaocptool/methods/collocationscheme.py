# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from casadi import DM, MX, repmat, vertcat, Function, jacobian, \
    vec, collocation_points, horzcat, substitute
from discretizationschemebase import DiscretizationSchemeBase


class CollocationScheme(DiscretizationSchemeBase):
    @property
    def time_interpolation(self):
        tau_list = self.solution_method.collocation_points(self.degree, with_zero=True)
        return [[t + self.solution_method.delta_t * tau for tau in tau_list] for t in self.time_breakpoints[:-1]]

    @property
    def time_interpolation_states(self):
        return self.time_interpolation

    @property
    def time_interpolation_algebraics(self):
        tau_list = self.solution_method.collocation_points(self.degree, with_zero=False)
        return [[t + self.solution_method.delta_t * tau for tau in tau_list] for t in self.time_breakpoints[:-1]]

    @property
    def time_interpolation_controls(self):
        tau_list = [0.] if self.degree_control == 1 else self.solution_method.collocation_points(self.degree,
                                                                                                 with_zero=False)
        return [[t + self.solution_method.delta_t * tau for tau in tau_list] for t in self.time_breakpoints[:-1]]

    def number_of_variables(self):
        return self.model.Nx * (self.finite_elements) * (self.degree + 1) \
               + self.model.Nyz * self.finite_elements * self.degree \
               + self.model.Nu * self.finite_elements * self.degree_control \
               + self.problem.N_eta

    def _create_nlp_symbolic_variables_and_bound_vectors(self):
        eta = MX.sym('eta', self.problem.N_eta)
        x = []
        y = []
        u = []
        vars_lb = []
        vars_ub = []
        for k in range(self.finite_elements):
            x_k = []
            for n in range(self.degree + 1):
                x_k.append(MX.sym('x_' + repr(k) + '_' + repr(n), self.model.Nx))
                vars_lb.append(self.problem.x_min)
                vars_ub.append(self.problem.x_max)
            x.append(x_k)
        for k in range(self.finite_elements):
            y_k = []
            for n in range(self.degree):
                y_k.append(MX.sym('yz_' + repr(k) + '_' + repr(n), self.model.Nyz))
                vars_lb.append(self.problem.yz_min)
                vars_ub.append(self.problem.yz_max)
            y.append(y_k)
        for k in range(self.finite_elements):
            u_k = []
            for n in range(self.degree_control):
                u_k.append(MX.sym('u_' + repr(k) + '_' + repr(n), self.model.Nu))
                vars_lb.append(self.problem.u_min)
                vars_ub.append(self.problem.u_max)
            u.append(vertcat(*u_k))

        v_x = vertcat(*[vertcat(*x_k) for x_k in x])
        v_y = vertcat(*[vertcat(*yz_k) for yz_k in y])
        v_u = vertcat(*u)
        v = vertcat(v_x, v_y, v_u, eta)

        vars_lb = vertcat(*vars_lb)
        vars_ub = vertcat(*vars_ub)

        return v, x, y, u, eta, vars_lb, vars_ub

    def splitXYandU(self, results_vector, all_subinterval=False):
        """
        :param all_subinterval: Bool 'Returns all elements of the subinterval (or only the first one)'
        :param results_vector: DM
        :return: X, Y, and U -> list with a DM for each element
        """
        assert (results_vector.__class__ == DM)
        x = []
        y = []
        u = []
        v_offset = 0
        if self.problem.N_eta > 0:
            results_vector = results_vector[:-self.problem.N_eta]

        for k in range(self.finite_elements):
            x_k = []
            for i in range(self.degree + 1):
                x_k.append(results_vector[v_offset:v_offset + self.model.Nx])
                v_offset += self.model.Nx
            if all_subinterval:
                x.append(x_k)
            else:
                x.append(x_k[0])
        x.append(x_k[-1])

        for k in range(self.finite_elements):
            y_k = []
            for i in range(self.degree):
                y_k.append(results_vector[v_offset:v_offset + self.model.Nyz])
                v_offset += self.model.Nyz
            if all_subinterval:
                y.append(y_k)
            else:
                y.append(y_k[0])

        for k in range(self.finite_elements):
            u_k = results_vector[v_offset:v_offset + self.model.Nu * self.degree_control]
            v_offset += self.model.Nu * self.degree_control
            u.append(u_k)
        assert v_offset == results_vector.numel()

        return x, y, u

    def discretize(self, finite_elements=None, degree=None, x_0=None, p=None, theta=None):
        if p is None:
            p = []

        if finite_elements is None:
            finite_elements = self.finite_elements
        if degree is None:
            degree = self.degree
        if theta is None:
            theta = dict([(i, []) for i in range(finite_elements)])
        if x_0 is None:
            x_0 = self.problem.x_0

        t0 = self.problem.t_0
        tf = self.problem.t_f

        v, x, yz, u, eta, vars_lb, vars_ub = self._create_nlp_symbolic_variables_and_bound_vectors()

        G = []

        F_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])

        ###
        u_pol, u_par = self.solution_method.createVariablePolynomialApproximation(self.model.Nu, self.degree_control,
                                                                                  name='u_col',
                                                                                  point_at_t0=False)

        tau, ell_list = self.solution_method.createLagrangianPolynomialBasis(self.degree, starting_index=0)
        d_ell_list = [jacobian(l, tau) for l in ell_list]
        f_d_ell_list = Function('f_dL_list', [tau], d_ell_list)

        G.append(F_h_initial(x[0][0], x_0))
        for n_element in range(self.finite_elements):
            dt = self.delta_t
            t_0_element = dt * n_element
            t_f_element = dt * (n_element + 1)

            tau_list = [0] + collocation_points(self.degree)
            mapping = lambda tau: (t_0_element + tau * dt)
            micro_t_k = map(mapping, tau_list)

            # self.model.convertFromTauToTime(dae_sys, t_0_element, t_f_element)

            if not n_element == 0:
                G.append(x[n_element - 1][-1] - x[n_element][0])

            for c_point in range(self.degree):
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

                p_i = vertcat(p, theta[n_element], u[n_element])

                arg = [tau_list[c_point + 1], x[n_element][c_point + 1], yz[n_element][c_point], p_i]
                # f_x_arg = [micro_t_k[i + 1], vec(horzcat(*X[n_element]))]

                d_x_d_tau = sum([f_d_ell_list(tau_list[c_point + 1])[k] * x[n_element][k] for k in range(degree + 1)])

                G.append(d_x_d_tau - dt * f_ode(*arg))
                G.append(f_alg(*arg))
            # XF = f_x_pol(*f_x_arg)
            XF = x[n_element][-1]

        G.append(f_h_final(XF, eta))

        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(XF, p)
        else:
            cost = 0

        nlp_prob = {}
        nlp_call = {}
        nlp_prob['g'] = vertcat(*G)
        nlp_prob['x'] = v
        #            nlp_prob['f'] = cost
        nlp_prob['f'] = cost
        nlp_call['lbx'] = vars_lb
        nlp_call['ubx'] = vars_ub
        nlp_call['lbg'] = DM.zeros(nlp_prob['g'].shape)
        nlp_call['ubg'] = DM.zeros(nlp_prob['g'].shape)

        return nlp_prob, nlp_call

    def create_initial_guess(self):
        base_x0 = repmat(self.problem.x_0, (self.degree + 1) * self.finite_elements)
        base_x0 = vertcat(base_x0,
                          DM.zeros((
                              self.model.Nyz * self.degree * self.finite_elements + self.model.Nu * self.degree_control * self.finite_elements),
                              1))

        base_x0 = vertcat(base_x0, DM.zeros(self.problem.N_eta))
        return base_x0

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """

        optimization_result.raw_solution_dict = raw_solution_dict
        optimization_result.raw_decision_variables = raw_solution_dict['x']

        raw_data = raw_solution_dict['x']
        x_breakpoints_values, y_breakpoints_values, u_breakpoints_values = self.splitXYandU(raw_data,
                                                                                            all_subinterval=False)
        x_interpolation_values, y_interpolation_values, u_interpolation_values = self.splitXYandU(raw_data,
                                                                                                  all_subinterval=True)

        optimization_result.objective = raw_solution_dict['f']
        optimization_result.constraint_values = raw_solution_dict['g']

        optimization_result.x_breakpoints_data['values'] = x_breakpoints_values
        optimization_result.y_breakpoints_data['values'] = y_breakpoints_values
        optimization_result.u_breakpoints_data['values'] = u_breakpoints_values

        optimization_result.x_breakpoints_data['time'] = self.time_breakpoints
        optimization_result.y_breakpoints_data['time'] = self.time_breakpoints[:-1]
        optimization_result.u_breakpoints_data['time'] = self.time_breakpoints[:-1]

        optimization_result.x_interpolation_data['values'] = x_interpolation_values
        optimization_result.y_interpolation_data['values'] = y_interpolation_values
        optimization_result.u_interpolation_data['values'] = u_interpolation_values

        optimization_result.x_interpolation_data['time'] = self.time_interpolation_states
        optimization_result.y_interpolation_data['time'] = self.time_interpolation_algebraics
        optimization_result.u_interpolation_data['time'] = self.time_interpolation_controls

        # def solve(self, initial_guess=None, p=[]):
        #     V_sol = self.solve_raw(initial_guess, p)
        #
        #     X, U = self.splitXandU(V_sol)
        #     X_finite_element = []
        #     U_finite_element = []
        #
        #     for x in X:
        #         X_finite_element.append(x[0])
        #     X_finite_element.append(X[-1][-1])
        #     #            for u in U:
        #     #                U_finite_element.append(u[0])
        #
        #     return X_finite_element, U, V_sol
