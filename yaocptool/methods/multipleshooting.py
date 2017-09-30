# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from casadi import DM, MX, vertcat, Function, repmat
from discretizationschemebase import DiscretizationSchemeBase


class MultipleShootingScheme(DiscretizationSchemeBase):
    def _number_of_variables(self):
        return self.model.Nx * (self.finite_elements + 1) \
               + self.finite_elements * self.model.Nu * self.degree_control \
               + self.problem.N_eta

    def _create_nlp_symbolic_variables_and_bound_vectors(self):
        n_v = self._number_of_variables()

        v = MX.sym("V", n_v)
        vars_lb = -DM.inf(n_v)
        vars_ub = DM.inf(n_v)

        x, u = self.splitXandU(v)

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
        eta = v[n_v - self.problem.N_eta:]
        return v, x, u, eta, vars_lb, vars_ub

    def splitXYandU(self, results_vector, all_subinterval=False):
        x = []
        y = []
        u = []

        v_offset = 0
        if self.problem.N_eta > 0:
            results_vector = results_vector[:-self.problem.N_eta]

        for k in range(self.finite_elements + 1):
            x.append(results_vector[v_offset:v_offset + self.model.Nx])
            v_offset = v_offset + self.model.Nx
            if k != self.finite_elements:
                u.append(results_vector[v_offset:v_offset + self.model.Nu * self.degree_control])
                y.append(DM([]))
                v_offset = v_offset + self.model.Nu * self.degree_control
        return x, y, u

    def discretize(self, finite_elements=None, x_0=None, p=None, theta=None):
        # TODO: Extract the G generation to another function
        if p is None:
            p = []
        finite_elements = self.finite_elements

        if theta is None:
            theta = dict([(i, []) for i in range(finite_elements)])

        if x_0 is None:
            x_0 = self.problem.x_0

        # Get the state at each shooting node
        all_decision_vars, x, u, eta, vars_lb, vars_ub = self._create_nlp_symbolic_variables_and_bound_vectors()
        constraint_list = []

        f_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])

        constraint_list.append(f_h_initial(x[0], x_0))

        for k in range(finite_elements):
            dae_sys = self.model.getDAESystem()
            self.model.convertFromTauToTime(dae_sys, self.time_breakpoints[k], self.time_breakpoints[k + 1])

            p_i = vertcat(p, theta[k], u[k])

            x_f = self.model.simulateStep(x[k], t_0=self.time_breakpoints[k], t_f=self.time_breakpoints[k + 1], p=p_i,
                                          dae_sys=dae_sys,
                                          integrator_type=self.solution_method.integrator_type)

            constraint_list.append(x_f - x[k + 1])

        constraint_list.append(f_h_final(x[-1], eta))

        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(x[-1], p)
        else:
            cost = 0

        nlp_prob = {'g': vertcat(*constraint_list),
                    'x': all_decision_vars,
                    'f': cost}

        lbg = DM.zeros(nlp_prob['g'].shape)
        ubg = DM.zeros(nlp_prob['g'].shape)

        nlp_call = {'lbx': vars_lb,
                    'ubx': vars_ub,
                    'lbg': lbg,
                    'ubg': ubg}

        return nlp_prob, nlp_call

    def create_initial_guess(self):
        base_x0 = self.problem.x_0
        base_x0 = vertcat(base_x0, repmat(DM([0] * self.model.Nu), self.degree_control))
        x0 = vertcat(repmat(base_x0, self.finite_elements), self.problem.x_0)
        x0 = vertcat(x0, DM.zeros(self.problem.N_eta))
        return x0

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """

        optimization_result.raw_solution_dict = raw_solution_dict
        optimization_result.raw_decision_variables = raw_solution_dict['x']

        optimization_result.objective = raw_solution_dict['f']
        optimization_result.constraint_values = raw_solution_dict['g']

        x_breakpoints_values, y_breakpoints_values, u_breakpoints_values = self.splitXYandU(raw_solution_dict['x'])
        optimization_result.x_breakpoints_data['values'] = x_breakpoints_values
        optimization_result.y_breakpoints_data['values'] = y_breakpoints_values
        optimization_result.u_breakpoints_data['values'] = u_breakpoints_values

        optimization_result.x_breakpoints_data['time'] = self.time_breakpoints
        optimization_result.y_breakpoints_data['time'] = self.time_breakpoints[:-1]
        optimization_result.u_breakpoints_data['time'] = self.time_breakpoints[:-1]

        optimization_result.x_interpolation_data['values'] = [val for val in x_breakpoints_values]
        optimization_result.y_interpolation_data['values'] = [val for val in y_breakpoints_values]
        optimization_result.u_interpolation_data['values'] = [val for val in u_breakpoints_values]

        optimization_result.x_interpolation_data['time'] = [t for t in self.time_breakpoints]
        optimization_result.y_interpolation_data['time'] = [t for t in self.time_breakpoints[:-1]]
        optimization_result.u_interpolation_data['time'] = [t for t in self.time_breakpoints[:-1]]
