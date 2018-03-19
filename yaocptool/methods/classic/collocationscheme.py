# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from collections import defaultdict
# noinspection PyUnresolvedReferences
from typing import List, Dict
from casadi import DM, MX, repmat, vertcat, Function, jacobian, is_equal, inf
from itertools import chain
from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase


# TODO: implement cost_as_a_sum

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

    def _number_of_variables(self):
        return self.model.n_x * self.finite_elements * (self.degree + 1) \
               + self.model.n_yz * self.finite_elements * self.degree \
               + self.model.n_u * self.finite_elements * self.degree_control \
               + self.problem.n_eta

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

        vars_lb.append(-DM.inf(self.problem.n_eta))
        vars_ub.append(DM.inf(self.problem.n_eta))

        vars_lb.append(self.problem.p_opt_min)
        vars_ub.append(self.problem.p_opt_max)

        vars_lb = vertcat(*vars_lb)
        vars_ub = vertcat(*vars_ub)
        return vars_lb, vars_ub

    def create_nlp_symbolic_variables(self):
        """
        Create the symbolic variables that will be used by the NLP problem
        :rtype: (MX, List[List[MX]], List[List[MX]], List[MX], MX, MX)
        """
        x_var = []
        y_var = []
        u_var = []
        for k in range(self.finite_elements):
            x_k = []
            for n in range(self.degree + 1):
                x_k.append(MX.sym('x_' + repr(k) + '_' + repr(n), self.model.n_x))
            x_var.append(x_k)

        for k in range(self.finite_elements):
            y_k = []
            for n in range(self.degree):
                y_k.append(MX.sym('yz_' + repr(k) + '_' + repr(n), self.model.n_yz))
            y_var.append(y_k)

        for k in range(self.finite_elements):
            u_k = []
            for n in range(self.degree_control):
                u_k.append(MX.sym('u_' + repr(k) + '_' + repr(n), self.model.n_u))
            u_var.append(u_k)

        eta = MX.sym('eta', self.problem.n_eta)
        p_opt = MX.sym('p_opt', self.problem.n_p_opt)

        v_x = self.vectorize(x_var)
        v_y = self.vectorize(y_var)
        v_u = self.vectorize(u_var)
        v = vertcat(v_x, v_y, v_u, eta, p_opt)

        return v, x_var, y_var, u_var, eta, p_opt

    def split_x_y_and_u(self, decision_variables, all_subinterval=False):
        """
        :param all_subinterval: Bool 'Returns all elements of the subinterval (or only the first one)'
        :param decision_variables: DM
        :return: X, Y, and U -> list with a DM for each element
        """

        x, y, u, _, _ = self.unpack_decision_variables(decision_variables, all_subinterval=all_subinterval)
        return x, y, u

    def unpack_decision_variables(self, decision_variables, all_subinterval=True):
        """Return a structured data from the decision variables vector

        Returns:
        (x_data, y_data, u_data, p_opt, eta)

        :param decision_variables: DM
        :param all_subinterval: bool
        :return: tuple
        """
        x, y, u, eta, p_opt = [], [], [], [], []
        v_offset = 0

        x_k = []
        for k in range(self.finite_elements):
            x_k = []
            for i in range(self.degree + 1):
                x_k.append(decision_variables[v_offset:v_offset + self.model.n_x])
                v_offset += self.model.n_x
            if all_subinterval:
                x.append(x_k)
            else:
                x.append(x_k[0])
        x.append([x_k[-1]])

        for k in range(self.finite_elements):
            y_k = []
            for i in range(self.degree):
                y_k.append(decision_variables[v_offset:v_offset + self.model.n_yz])
                v_offset += self.model.n_yz
            if all_subinterval:
                y.append(y_k)
            else:
                y.append(y_k[0])

        for k in range(self.finite_elements):
            u_k = []
            for i in range(self.degree_control):
                u_k.append(decision_variables[v_offset:v_offset + self.model.n_u])
                v_offset += self.model.n_u
            u.append(u_k)

        if self.problem.n_eta > 0:
            eta = decision_variables[v_offset:v_offset + self.problem.n_eta]
            v_offset += self.problem.n_eta

        if self.problem.n_p_opt > 0:
            p_opt = decision_variables[v_offset:v_offset + self.problem.n_p_opt]
            v_offset += self.problem.n_p_opt

        assert v_offset == decision_variables.numel()

        return x, y, u, eta, p_opt

    def discretize(self, x_0=None, p=None, theta=None, last_u=None):
        if p is None:
            p = []
        if theta is None:
            theta = dict([(i, []) for i in range(self.finite_elements)])
        if x_0 is None:
            x_0 = self.problem.x_0

        # Create NLP symbolic variables
        all_decision_vars, x_var, yz_var, u_var, eta, p_opt = self.create_nlp_symbolic_variables()

        cost = 0
        constraint_list = []
        lbg = []
        ubg = []

        # Put the symbolic optimization parameters in the parameter vector
        for i, p_opt_index in enumerate(self.problem.get_p_opt_indices()):
            p[p_opt_index] = p_opt[i]

        # Create "simulations" time_dict, a dict informing the simulation points in each finite elem. for each variable
        time_dict = self._create_time_dict_for_collocation()

        # create a polynomial approximation for x
        x_pol, x_par = self.solution_method.create_variable_polynomial_approximation(self.model.n_x, self.degree,
                                                                                     name='col_x_approx',
                                                                                     point_at_t0=True)

        func_d_x_pol_d_tau = Function('f_dL_list', [self.model.tau_sym, x_par],
                                      [jacobian(x_pol, self.model.tau_sym)])

        # Initial time constraint/initial condition
        f_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        constraint_list.append(f_h_initial(x_var[0][0], x_0))
        lbg.append(DM.zeros(constraint_list[-1].shape))
        ubg.append(DM.zeros(constraint_list[-1].shape))

        # Create functions to be evaluated
        functions = defaultdict(dict)
        for el in range(self.finite_elements):
            dae_sys = self.model.get_dae_system()

            dae_sys.convert_from_tau_to_time(self.time_breakpoints[el],
                                             self.time_breakpoints[el + 1])

            f_ode = Function('f_ode_' + repr(el), self.model.all_sym, [dae_sys.ode])
            f_alg = Function('f_alg_' + repr(el), self.model.all_sym, [dae_sys.alg])

            functions['ode'][el] = f_ode
            functions['alg'][el] = f_alg

        # Obtain the "simulation" results
        results = self.get_system_at_given_times(x_var, yz_var, u_var, time_dict, p, theta, functions=functions)

        for el in range(self.finite_elements):
            dt = self.solution_method.delta_t

            # State continuity, valid for all but the first finite element
            if not el == 0:
                constraint_list.append(results[el - 1]['x'][-1] - results[el]['x'][0])
                lbg.append(DM.zeros(constraint_list[-1].shape))
                ubg.append(DM.zeros(constraint_list[-1].shape))

            tau_list = self.solution_method.collocation_points(self.degree, with_zero=True)

            # Enforce the the derivative of the polynomial to be equal ODE at t
            for col_point in range(1, self.degree + 1):
                constraint_list.append(func_d_x_pol_d_tau(tau_list[col_point], self.vectorize(x_var[el]))
                                       - dt * results[el]['ode'][col_point])
                lbg.append(DM.zeros(constraint_list[-1].shape))
                ubg.append(DM.zeros(constraint_list[-1].shape))

            for col_point in range(self.degree):
                constraint_list.append(results[el]['alg'][col_point])
                lbg.append(DM.zeros(constraint_list[-1].shape))
                ubg.append(DM.zeros(constraint_list[-1].shape))

            # Implement the constraint on delta_u
            if self.problem.has_delta_u:
                if el > 0:
                    for i in range(self.model.n_u):
                        if not is_equal(self.problem.delta_u_max[i], inf) or not is_equal(self.problem.delta_u_min[i],
                                                                                          -inf):
                            constraint_list.append(u_var[el][0][i] - u_var[el - 1][0][i])
                            lbg.append(self.problem.delta_u_min[i])
                            ubg.append(self.problem.delta_u_max[i])

                elif el == 0 and last_u is not None:
                    for i in range(self.model.n_u):
                        if not is_equal(self.problem.delta_u_max[i], inf) or not is_equal(self.problem.delta_u_min[i],
                                                                                          -inf):
                            constraint_list.append(u_var[el][0][i] - last_u[i])
                            lbg.append(self.problem.delta_u_min[i])
                            ubg.append(self.problem.delta_u_max[i])

        # Final constraint
        x_f = results[self.finite_elements - 1]['x'][-1]
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta, self.model.p_sym], [self.problem.h_final])
        constraint_list.append(f_h_final(x_f, eta, p))
        lbg.append(DM.zeros(constraint_list[-1].shape))
        ubg.append(DM.zeros(constraint_list[-1].shape))

        # Time independent constraints
        f_h = Function('h', [self.model.p_sym], [self.problem.h])
        constraint_list.append(f_h(p))
        lbg.append(DM.zeros(constraint_list[-1].shape))
        ubg.append(DM.zeros(constraint_list[-1].shape))

        # Cost function
        if self.solution_method.solution_class == 'direct':
            cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])(x_f, p)
        else:
            cost = 0

        nlp_prob = {'g': vertcat(*constraint_list),
                    'x': all_decision_vars,
                    'f': cost}

        lbg = vertcat(*lbg)
        ubg = vertcat(*ubg)

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
        :param u: List[List[MX]]
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
            u_func = self.model.convert_expr_from_tau_to_time(self.model.u_func, t_0, t_f)
            if self.solution_method.solution_class == 'direct':
                f_u = Function('f_u_pol', [self.model.t_sym, self.model.u_par], [u_func])
            else:
                f_u = Function('f_u_pol', list(self.model.all_sym), [u_func])

            # Create function for obtaining x at an given time
            x_pol, x_par = self.solution_method.create_variable_polynomial_approximation(self.model.n_x, self.degree,
                                                                                         name='col_x_approx',
                                                                                         point_at_t0=True)
            x_pol = self.model.convert_expr_from_tau_to_time(x_pol, t_k=t_0, t_kp1=t_f)
            f_x = Function('f_x_pol', [self.model.t_sym, x_par], [x_pol])

            # Create function for obtaining y at an given time
            y_pol, y_par = self.solution_method.create_variable_polynomial_approximation(self.model.n_yz, self.degree,
                                                                                         name='col_y_approx',
                                                                                         point_at_t0=False)
            y_pol = self.model.convert_expr_from_tau_to_time(y_pol, t_k=t_0, t_kp1=t_f)
            f_y = Function('f_y_pol', [self.model.t_sym, y_par], [y_pol])

            # Find the times that need to be evaluated
            element_breakpoints = set()
            for key in chain(['x', 'y', 'u'], functions.keys()):
                if key in time_dict[el]:
                    element_breakpoints = element_breakpoints.union(time_dict[el][key])

            element_breakpoints = list(element_breakpoints)
            element_breakpoints.sort()

            # Iterate with the times in the finite element
            for t in element_breakpoints:
                x_t = f_x(t, self.vectorize(x[el]))
                yz_t = f_y(t, self.vectorize(y[el]))
                y_t, z_t = self.model.slice_yz_to_y_and_z(yz_t)

                if self.solution_method.solution_class == 'direct':
                    u_t = f_u(t, self.vectorize(u[el]))
                else:
                    u_t = f_u(*self.model.put_values_in_all_sym_format(t, x=x_t, y=y_t, z=z_t, p=p,
                                                                       theta=theta[el],
                                                                       u_par=self.vectorize(u[el])))

                if 'x' in time_dict[el] and t in time_dict[el]['x']:
                    results[el]['x'].append(x_t)
                if 'y' in time_dict and t in time_dict[el]['y']:
                    results[el]['y'].append(yz_t)
                if 'u' in time_dict and t in time_dict[el]['u']:
                    results[el]['u'].append(u_t)

                for f_name in functions:
                    if t in time_dict[el][f_name]:
                        f = functions[f_name][el]
                        val = f(*self.model.put_values_in_all_sym_format(t=t, x=x_t, y=y_t, z=z_t, p=p, theta=theta[el],
                                                                         u_par=self.vectorize(u[el])))
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
        x_init = repmat(self.problem.x_0, (self.degree + 1) * self.finite_elements)
        if self.problem.y_guess is not None:
            y_init = repmat(self.problem.y_guess, self.degree * self.finite_elements)
        else:
            y_init = repmat(DM.zeros(self.model.n_y), self.degree * self.finite_elements)

        if self.problem.u_guess is not None:
            u_init = repmat(self.problem.u_guess, self.degree_control * self.finite_elements)
        else:
            u_init = repmat(DM.zeros(self.model.n_u), self.degree_control * self.finite_elements)

        eta_init = DM.zeros(self.problem.n_eta, 1)
        p_init = DM.zeros(self.problem.n_p_opt, 1)

        return vertcat(x_init, y_init, u_init, eta_init, p_init)

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :type optimization_result: yaocptool.methods.base.optimizationresult.OptimizationResult
        :type raw_solution_dict: dict
        """

        optimization_result.raw_solution_dict = raw_solution_dict
        optimization_result.raw_decision_variables = raw_solution_dict['x']

        raw_data = raw_solution_dict['x']
        x_interpolation_values, y_interpolation_values, u_interpolation_values \
            = self.split_x_y_and_u(raw_data, all_subinterval=True)

        x_values, y_values, u_values, eta, p_opt = self.unpack_decision_variables(raw_solution_dict['x'])

        optimization_result.p_opt = p_opt
        optimization_result.eta = eta

        optimization_result.objective = raw_solution_dict['f']
        optimization_result.constraint_values = raw_solution_dict['g']

        optimization_result.x_interpolation_data['values'] = x_interpolation_values
        optimization_result.y_interpolation_data['values'] = y_interpolation_values
        optimization_result.u_interpolation_data['values'] = u_interpolation_values

        optimization_result.x_interpolation_data['time'] = self.time_interpolation_states
        optimization_result.y_interpolation_data['time'] = self.time_interpolation_algebraics
        optimization_result.u_interpolation_data['time'] = self.time_interpolation_controls
