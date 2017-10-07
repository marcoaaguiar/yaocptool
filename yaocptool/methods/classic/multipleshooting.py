# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from collections import defaultdict

from casadi import DM, MX, vertcat, Function, repmat
# noinspection PyUnresolvedReferences
from typing import Dict, List
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase


class MultipleShootingScheme(DiscretizationSchemeBase):
    def _number_of_variables(self):
        return self.model.n_x * (self.finite_elements + 1) \
               + self.finite_elements * self.model.n_u * self.degree_control \
               + self.problem.n_eta

    def _create_variables_bound_vectors(self):
        """
        Return two items: the vector of lower bounds and upper bounds
        :rtype: (DM, DM)
        """
        vars_lb = []
        vars_ub = []
        for k in range(self.finite_elements + 1):
            vars_lb.append(self.problem.x_min)
            vars_ub.append(self.problem.x_max)
        # for k in range(self.finite_elements):
        #         vars_lb.append(self.problem.yz_min)
        #         vars_ub.append(self.problem.yz_max)
        for k in range(self.finite_elements):
            for j in range(self.degree_control):
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
        eta = MX.sym('eta', self.problem.n_eta)
        x_var, y_var, u_var = [], [], []

        for el in range(self.finite_elements + 1):
            x_k = [MX.sym('x_' + repr(el), self.model.n_x)]
            x_var.append(x_k)

        for el in range(self.finite_elements):
            # y_k = [MX.sym('yz_' + repr(el), self.model.n_yz)]
            y_k = []
            y_var.append(y_k)

        for el in range(self.finite_elements):
            u_k = []
            for n in range(self.degree_control):
                u_k.append(MX.sym('u_' + repr(el) + '_' + repr(n), self.model.n_u))
            u_var.append(u_k)

        v_x = self.vectorize(x_var)
        v_y = self.vectorize(y_var)
        v_u = self.vectorize(u_var)
        v = vertcat(v_x, v_y, v_u, eta)

        return v, x_var, y_var, u_var, eta

    def split_x_y_and_u(self, results_vector, all_subinterval=False):
        """
        :param all_subinterval: Bool 'Returns all elements of the subinterval (or only the first one)'
        :param results_vector: DM
        :return: X, Y, and U -> list with a DM for each element
        """
        assert (results_vector.__class__ == DM)
        x, y, u = [], [], []
        v_offset = 0

        if self.problem.n_eta > 0:
            results_vector = results_vector[:-self.problem.n_eta]

        for k in range(self.finite_elements + 1):
            x.append([results_vector[v_offset:v_offset + self.model.n_x]])
            v_offset = v_offset + self.model.n_x

        for k in range(self.finite_elements):
                y.append([DM([])])

        for k in range(self.finite_elements):
            u_k = []
            for i in range(self.degree_control):
                u_k.append(results_vector[v_offset:v_offset + self.model.n_u])
                v_offset += self.model.n_u
            u.append(u_k)

        assert v_offset == results_vector.numel()

        return x, y, u

    def discretize(self, x_0=None, p=None, theta=None):
        if p is None:
            p = []

        if theta is None:
            theta = dict([(i, []) for i in range(self.finite_elements)])

        if x_0 is None:
            x_0 = self.problem.x_0

        # Create NLP symbolic variables
        all_decision_vars, x_var, yz_var, u_var, eta = self.create_nlp_symbolic_variables()
        y = []
        constraint_list = []

        # Create "simulations" time_dict, a dict informing the simulation points in each finite elem. for each variable
        time_dict = self._create_time_dict_for_multiple_shooting()

        # Initial time constraint/initial condition
        f_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        constraint_list.append(f_h_initial(x_var[0][0], x_0))

        # Multiple Shooting "simulation"
        results = self.get_system_at_given_times(x_var, y, u_var, time_dict, p, theta)
        for el in range(self.finite_elements):
            constraint_list.append(results[el]['x'][0] - x_var[el + 1][0])

        # Final time constraint
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta], [self.problem.h_final])
        constraint_list.append(f_h_final(x_var[-1][0], eta))

        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(x_var[-1][0], p)
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

    def _create_time_dict_for_multiple_shooting(self):
        time_dict = defaultdict(dict)
        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['x'] = [self.time_breakpoints[el + 1]]

        return time_dict

    def get_system_at_given_times(self, x_var, y_var, u_var, time_dict=None, p=None, theta=None, functions=None,
                                  start_at_t_0=False):
        """
        :param x_var: List[List[MX]]
        :param y_var: List[List[MX]]
        :param u_var: List[List[MX]]
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
        # TODO make the results[el]['x'] be indexed by the evaluated time (a dict where the key is t) instead of list

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
            t_0 = t_init = time_dict[el]['t_0']
            t_f = time_dict[el]['t_f']
            x_init = x_var[el][0]
            # Create dae_sys and the control function
            dae_sys = self.model.get_dae_system()
            self.model.convert_dae_sys_from_tau_to_time(dae_sys, self.time_breakpoints[el],
                                                        self.time_breakpoints[el + 1])
            u_func = self.model.convert_expr_from_tau_to_time(self.model.u_func, t_0, t_f)
            if self.solution_method.solution_class == 'direct':
                f_u = Function('f_u_pol', [self.model.t_sym, self.model.u_par], [u_func])
            else:
                f_u = Function('f_u_pol', list(self.model.all_sym), [u_func])

            # Find the times that need to be evaluated
            element_breakpoints = set()
            for key in ['x', 'y', 'u'] + functions.keys():
                if key in time_dict[el]:
                    element_breakpoints = element_breakpoints.union(time_dict[el][key])

            element_breakpoints = list(element_breakpoints)
            element_breakpoints.sort()

            # If values are needed from t_0, get it
            if 'x' in time_dict[el] and t_0 in time_dict[el]['x']:
                results[el]['x'].append(x_var[el][0])
            if 'y' in time_dict[el] and t_0 in time_dict[el]['y']:
                raise NotImplementedError
            if 'u' in time_dict[el] and t_0 in time_dict[el]['u']:
                if self.solution_method.solution_class == 'direct':
                    results[el]['x'].append(f_u(t_0, self.vectorize(u_var[el])))
                else:
                    raise NotImplementedError
            for f_name in functions:
                if t_0 in time_dict[el][f_name]:
                    raise NotImplementedError

            # Remove t_0 from the list of times that need to be evaluated
            if t_0 in element_breakpoints:
                element_breakpoints.remove(t_0)

            for t in element_breakpoints:
                t_next = t
                p_i = vertcat(p, theta[el], self.vectorize(u_var[el]))

                # Do the simulation
                sim_result = self.model.simulate_step(x_init, t_0=t_init, t_f=t_next, p=p_i,
                                                      dae_sys=dae_sys,
                                                      integrator_type=self.solution_method.integrator_type)

                # Fetch values from results
                x_t, yz_t = sim_result['xf'], sim_result['zf']
                y_t, z_t = self.model.slice_yz_to_y_and_z(yz_t)

                # Save to the result vector
                if 'x' in time_dict[el] and t in time_dict[el]['x']:
                    results[el]['x'].append(x_t)
                if 'y' in time_dict and t in time_dict[el]['y']:
                    results[el]['y'].append(yz_t)
                if 'u' in time_dict and t in time_dict[el]['u']:
                    if self.solution_method.solution_class == 'direct':
                        results[el]['u'].append(f_u(t, self.vectorize(u_var[el])))
                    else:
                        results[el]['u'].append(
                            f_u(*self.model.put_values_in_all_sym_format(t, x=x_t, y=y_t, z=z_t, p=p,
                                                                         theta=theta[el],
                                                                         u_par=u_var[el])))
                for f_name in functions:
                    if t in time_dict[el][f_name]:
                        f = functions[f_name][el]
                        val = f(
                            *self.model.put_values_in_all_sym_format(t=t, x=x_t, y=yz_t, z=z_t, p=p, theta=theta[el],
                                                                     u_par=self.vectorize(u_var[el])))
                        results[el][f_name].append(val)
                # If the simulation should start from the begin of the simulation interval, do not change the t_init
                if not start_at_t_0:
                    t_init = t
                    x_init = x_t
        return results

    def create_initial_guess(self):
        base_x0 = self.problem.x_0
        base_x0 = vertcat(base_x0, repmat(DM([0] * self.model.n_u), self.degree_control))
        x0 = vertcat(repmat(base_x0, self.finite_elements), self.problem.x_0)
        x0 = vertcat(x0, DM.zeros(self.problem.n_eta))
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

        x_breakpoints_values, y_breakpoints_values, u_breakpoints_values = self.split_x_y_and_u(raw_solution_dict['x'])

        optimization_result.x_breakpoints_data['values'] = x_breakpoints_values
        optimization_result.y_breakpoints_data['values'] = y_breakpoints_values
        optimization_result.u_breakpoints_data['values'] = u_breakpoints_values

        optimization_result.x_breakpoints_data['time'] = self.time_breakpoints
        optimization_result.y_breakpoints_data['time'] = self.time_breakpoints[:-1]
        optimization_result.u_breakpoints_data['time'] = self.time_breakpoints[:-1]

        optimization_result.x_interpolation_data['values'] = x_breakpoints_values
        optimization_result.y_interpolation_data['values'] = y_breakpoints_values
        optimization_result.u_interpolation_data['values'] = y_breakpoints_values

        optimization_result.x_interpolation_data['time'] = [t for t in self.time_breakpoints]
        optimization_result.y_interpolation_data['time'] = [t for t in self.time_breakpoints[:-1]]
        optimization_result.u_interpolation_data['time'] = [t for t in self.time_breakpoints[:-1]]
