# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from collections import defaultdict
from typing import List, Dict
from casadi import DM, MX, repmat, vertcat, Function, jacobian
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase


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
        return self.model.Nx * self.finite_elements * (self.degree + 1) \
               + self.model.Nyz * self.finite_elements * self.degree \
               + self.model.Nu * self.finite_elements * self.degree_control \
               + self.problem.N_eta

    def _create_nlp_symbolic_variables_and_bound_vectors(self):
        """
        Create the symbolic variables that will be used by the NLP problem
        :rtype: (DM, List(List(DM)), List(List(DM)), List(DM), DM, DM, DM)
        """
        v, x, y, u, eta = self.create_nlp_symbolic_variables()
        vars_lb, vars_ub = self._create_variables_bound_vectors()

        return v, x, y, u, eta, vars_lb, vars_ub

    def _create_variables_bound_vectors(self):
        """
        Return two items: the vector of lower bounds and upperbounds
        :rtype: (DM, DM)
        """
        vars_lb = []
        vars_ub = []
        for k in range(self.finite_elements):
            for n in range(self.degree + 1):
                vars_lb.append(self.problem.x_min)
                vars_ub.append(self.problem.x_max)
        for k in range(self.finite_elements):
            for n in range(self.degree):
                vars_lb.append(self.problem.yz_min)
                vars_ub.append(self.problem.yz_max)
        for k in range(self.finite_elements):
            for n in range(self.degree_control):
                vars_lb.append(self.problem.u_min)
                vars_ub.append(self.problem.u_max)
        vars_lb = vertcat(*vars_lb)
        vars_ub = vertcat(*vars_ub)
        return vars_lb, vars_ub

    def create_nlp_symbolic_variables(self):
        """
        Create the symbolic variables that will be used by the NLP problem
        :rtype: (MX, List[List[MX]], List[List[MX]], List[MX], MX)
        """
        eta = MX.sym('eta', self.problem.N_eta)
        x = []
        y = []
        u = []
        for k in range(self.finite_elements):
            x_k = []
            for n in range(self.degree + 1):
                x_k.append(MX.sym('x_' + repr(k) + '_' + repr(n), self.model.Nx))
            x.append(x_k)

        for k in range(self.finite_elements):
            y_k = []
            for n in range(self.degree):
                y_k.append(MX.sym('yz_' + repr(k) + '_' + repr(n), self.model.Nyz))
            y.append(y_k)

        for k in range(self.finite_elements):
            u_k = []
            for n in range(self.degree_control):
                u_k.append(MX.sym('u_' + repr(k) + '_' + repr(n), self.model.Nu))
            u.append(vertcat(*u_k))

        v_x = vertcat(*[vertcat(*x_k) for x_k in x])
        v_y = vertcat(*[vertcat(*yz_k) for yz_k in y])
        v_u = vertcat(*u)
        v = vertcat(v_x, v_y, v_u, eta)

        return v, x, y, u, eta

    def splitXYandU(self, results_vector, all_subinterval=False):
        """
        :param all_subinterval: Bool 'Returns all elements of the subinterval (or only the first one)'
        :param results_vector: DM
        :return: X, Y, and U -> list with a DM for each element
        """
        assert (results_vector.__class__ == DM)
        x, y, u = [], [], []
        v_offset = 0

        if self.problem.N_eta > 0:
            results_vector = results_vector[:-self.problem.N_eta]

        x_k = []
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

    def discretize(self, x_0=None, p=None, theta=None):
        if p is None:
            p = []

        finite_elements = self.finite_elements
        if theta is None:
            theta = dict([(i, []) for i in range(finite_elements)])
        if x_0 is None:
            x_0 = self.problem.x_0

        # Create NLP symbolic variables
        all_decision_vars, x_var, yz_var, u_var, eta = self.create_nlp_symbolic_variables()
        constraint_list = []

        # Create "simulations" time_dict
        time_dict = self._create_time_dict_for_collocation()

        ###
        x_pol, x_par = self.solution_method.createVariablePolynomialApproximation(self.model.Nx, self.degree,
                                                                                  name='col_x_approx',
                                                                                  point_at_t0=True)

        func_d_x_pol_d_tau = Function('f_dL_list', [self.model.tau_sym, x_par],
                                      [jacobian(x_pol, self.model.tau_sym)])

        f_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        constraint_list.append(f_h_initial(x_var[0][0], x_0))

        # Create functions to be evaluated
        functions = defaultdict(dict)
        for el in range(self.finite_elements):
            dae_sys = self.model.getDAESystem()
            if 'z' not in dae_sys:
                dae_sys['z'] = vertcat([])
                dae_sys['alg'] = vertcat([])

            self.model.convert_dae_sys_from_tau_to_time(dae_sys, self.time_breakpoints[el],
                                                        self.time_breakpoints[el + 1])

            f_ode = Function('f_ode_' + repr(el), self.model.all_sym, [dae_sys['ode']])
            f_alg = Function('f_alg_' + repr(el), self.model.all_sym, [dae_sys['alg']])

            functions['ode'][el] = f_ode
            functions['alg'][el] = f_alg

        # Obtain the "simulation" results
        results = self.get_system_at_given_times(x_var, yz_var, u_var, time_dict, p, theta, functions=functions)

        for el in range(self.finite_elements):
            dt = self.solution_method.delta_t

            # State continuity, valid for all but the first finite element
            if not el == 0:
                constraint_list.append(results[el - 1]['x'][-1] - results[el]['x'][0])

            tau_list = self.solution_method.collocation_points(self.degree, with_zero=True)

            # Enforce the the derivative of the polynomial to be equal ODE at t
            for col_point in range(1, self.degree + 1):
                constraint_list.append(
                    func_d_x_pol_d_tau(tau_list[col_point], vertcat(*x_var[el])) - dt * results[el]['ode'][col_point])

            for col_point in range(self.degree):
                constraint_list.append(results[el]['alg'][col_point])

        # Final constraint
        x_f = results[self.finite_elements - 1]['x'][-1]
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])
        constraint_list.append(f_h_final(x_f, eta))

        # Cost function
        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(x_f, p)
        else:
            cost = 0

        nlp_prob = {'g': vertcat(*constraint_list),
                    'x': all_decision_vars,
                    'f': cost}

        lbg = DM.zeros(nlp_prob['g'].shape)
        ubg = DM.zeros(nlp_prob['g'].shape)

        vars_lb, vars_ub = self._create_variables_bound_vectors()
        nlp_call = {'lbx': vars_lb,
                    'ubx': vars_ub,
                    'lbg': lbg,
                    'ubg': ubg}

        return nlp_prob, nlp_call

    def get_system_at_given_times(self, x, y, u, time_dict=None, p=None, theta=None, functions=None,
                                  start_at_t_0=False):
        """
        :param x: List[List[MX]]
        :param y: List[List[MX]]
        :param u: List[MX]
        :type time_dict: Dict(int, List(float)) Dictionary of simulations times, where the KEY is the
                                                finite_element and the VALUE list a list of desired times
                                                example : {1:{'t_0': 0.0, 'x':[0.0, 0.1, 0.2], y:[0.2]}}
        :param p: list
        :param theta: dict
        :param start_at_t_0: bool If TRUE the simulations in each finite_element will start at the element t_0,
                                  Otherwise the simulation will start the end of the previous element
        :param functions: Dict[str, Function|Dict[int] dictionary of Functions to be evaluated, KEY is the function
                                              identifier, VALUE is a CasADi Function with model.all_sym as input
         """

        if theta is None:
            theta = {}
        if p is None:
            p = []
        if time_dict is None:
            time_dict = {}
        if functions is None:
            functions = {}
        results = defaultdict(lambda: defaultdict(list))

        # for el in range(finite_element):
        for el in time_dict:
            t_0 = time_dict[el]['t_0']
            t_f = time_dict[el]['t_f']

            # The control function
            u_func = self.model.convertExprFromTauToTime(self.model.u_func, t_0, t_f)
            if self.solution_method.solution_class == 'direct':
                f_u = Function('f_u_pol', [self.model.t_sym, self.model.u_par], [u_func])
            else:
                f_u = Function('f_u_pol', list(self.model.all_sym), [u_func])

            # Create function for obtaining x at an given time
            x_pol, x_par = self.solution_method.createVariablePolynomialApproximation(self.model.Nx, self.degree,
                                                                                      name='col_x_approx',
                                                                                      point_at_t0=True)
            x_pol = self.model.convertExprFromTauToTime(x_pol, t_k=t_0, t_kp1=t_f)
            f_x = Function('f_x_pol', [self.model.t_sym, x_par], [x_pol])

            # Create function for obtaining y at an given time
            y_pol, y_par = self.solution_method.createVariablePolynomialApproximation(self.model.Nyz, self.degree,
                                                                                      name='col_y_approx',
                                                                                      point_at_t0=False)
            y_pol = self.model.convertExprFromTauToTime(y_pol, t_k=t_0, t_kp1=t_f)
            f_y = Function('f_y_pol', [self.model.t_sym, y_par], [y_pol])

            # Find the times that need to be evaluated
            element_breakpoints = set()
            for key in ['x', 'y', 'u'] + functions.keys():
                if key in time_dict[el]:
                    element_breakpoints = element_breakpoints.union(time_dict[el][key])

            element_breakpoints = list(element_breakpoints)
            element_breakpoints.sort()

            # Iterate with the times in the finite element
            for t in element_breakpoints:
                x_t = f_x(t, vertcat(*x[el]))
                yz_t = f_y(t, vertcat(*y[el]))
                y_t, z_t = self.model.slice_yz_to_y_and_z(yz_t)

                if 'x' in time_dict[el] and t in time_dict[el]['x']:
                    results[el]['x'].append(x_t)
                if 'y' in time_dict and t in time_dict[el]['y']:
                    results[el]['y'].append(yz_t)
                if 'u' in time_dict and t in time_dict[el]['u']:
                    if self.solution_method.solution_class == 'direct':
                        results[el]['u'].append(f_u(t, u[el]))
                    else:
                        results[el]['u'].append(
                            f_u(*self.model.put_values_in_all_sym_format(t, x=x_t, y=y_t, z=z_t, p=p,
                                                                         theta=theta[el],
                                                                         u_par=u[el])))
                for f_name in functions:
                    if t in time_dict[el][f_name]:
                        f = functions[f_name][el]
                        val = f(*self.model.put_values_in_all_sym_format(t=t, x=x_t, y=y_t, z=z_t, p=p, theta=theta[el],
                                                                         u_par=u[el]))
                        results[el][f_name].append(val)
        return results

    def _create_time_dict_for_collocation(self):
        time_dict = defaultdict(dict)
        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['x'] = self.time_interpolation_states[el]
            time_dict[el]['y'] = self.time_interpolation_algebraics[el]
            time_dict[el]['u'] = self.time_interpolation_controls[el]

            time_dict[el]['ode'] = self.time_interpolation_states[el]
            time_dict[el]['alg'] = self.time_interpolation_algebraics[el]

        return time_dict

    def create_initial_guess(self):
        base_x0 = repmat(self.problem.x_0, (self.degree + 1) * self.finite_elements)
        base_x0 = vertcat(base_x0,
                          DM.zeros((
                              self.model.Nyz * self.degree * self.finite_elements +
                              self.model.Nu * self.degree_control * self.finite_elements),
                              1))

        base_x0 = vertcat(base_x0, DM.zeros(self.problem.N_eta))
        return base_x0

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.base.optimizationresult.OptimizationResult
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
