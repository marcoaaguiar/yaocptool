# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:08:34 2017

@author: marco
"""
from collections import defaultdict

from casadi import DM, vertcat, Function, repmat, is_equal, inf

from yaocptool.methods.base.discretizationschemebase import DiscretizationSchemeBase
from yaocptool.optimization import NonlinearOptimizationProblem


class MultipleShootingScheme(DiscretizationSchemeBase):
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

        for k in range(self.finite_elements):
            vars_lb.append(self.problem.y_min)
            vars_ub.append(self.problem.y_max)

        for k in range(self.finite_elements):
            for j in range(self.degree_control):
                vars_lb.append(self.problem.u_min)
                vars_ub.append(self.problem.u_max)

        vars_lb.append(-DM.inf(self.problem.n_eta))
        vars_ub.append(DM.inf(self.problem.n_eta))

        vars_lb.append(self.problem.p_opt_min)
        vars_ub.append(self.problem.p_opt_max)

        for el in range(self.finite_elements):
            vars_lb.append(self.problem.theta_opt_min)
            vars_ub.append(self.problem.theta_opt_max)

        vars_lb = vertcat(*vars_lb)
        vars_ub = vertcat(*vars_ub)
        return vars_lb, vars_ub

    def create_nlp_symbolic_variables(self, nlp):
        """
       Create the symbolic variables that will be used by the NLP problem
       :rtype: (MX, List[List[MX]], List[List[MX]], List[MX], MX, MX, List[MX)
       """
        x_var, y_var, u_var = [], [], []

        for el in range(self.finite_elements + 1):
            x_k = [nlp.create_variable('x_' + str(el), self.model.n_x, lb=self.problem.x_min, ub=self.problem.x_max)]
            x_var.append(x_k)

        for el in range(self.finite_elements):
            y_k = [nlp.create_variable('y_' + str(el), self.model.n_y, lb=self.problem.y_min, ub=self.problem.y_max)]
            y_var.append(y_k)

        for el in range(self.finite_elements):
            u_k = []
            for n in range(self.degree_control):
                u_k.append(nlp.create_variable('u_' + str(el) + '_' + str(n), self.model.n_u,
                                               lb=self.problem.u_min, ub=self.problem.u_max))
            u_var.append(u_k)

        eta = nlp.create_variable('eta', self.problem.n_eta)
        p_opt = nlp.create_variable('p_opt', self.problem.n_p_opt, lb=self.problem.p_opt_min, ub=self.problem.p_opt_max)

        theta_opt = []
        for el in range(self.finite_elements):
            theta_opt.append(nlp.create_variable('theta_opt_' + str(el), self.problem.n_theta_opt,
                                                 lb=self.problem.theta_opt_min, ub=self.problem.theta_opt_max))

        v_x = self.vectorize(x_var)
        v_y = self.vectorize(y_var)
        v_u = self.vectorize(u_var)
        v_theta_opt = vertcat(*theta_opt)

        v = vertcat(v_x, v_y, v_u, eta, p_opt, v_theta_opt)

        return v, x_var, y_var, u_var, eta, p_opt, theta_opt

    def split_x_y_and_u(self, results_vector, all_subinterval=False):
        """
        :param all_subinterval: Bool 'Returns all elements of the subinterval (or only the first one)'
        :param results_vector: DM
        :return: X, Y, and U -> list with a DM for each element
        """
        x, y, u, _, _, _ = self.unpack_decision_variables(results_vector)

        return x, y, u

    def unpack_decision_variables(self, decision_variables):
        """Return a structured data from the decision variables vector

        Returns:
        (x_data, y_data, u_data, p_opt, eta, theta_opt)

        :param decision_variables: DM
        :return: tuple
        """
        x, y, u, eta, p_opt = [], [], [], [], []
        v_offset = 0

        for k in range(self.finite_elements + 1):
            x.append([decision_variables[v_offset:v_offset + self.model.n_x]])
            v_offset = v_offset + self.model.n_x

        for k in range(self.finite_elements):
            y.append([decision_variables[v_offset:v_offset + self.model.n_y]])
            v_offset = v_offset + self.model.n_y

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

        theta_opt = []
        for k in range(self.finite_elements):
            theta_opt.append(decision_variables[v_offset:v_offset + self.problem.n_theta_opt])
            v_offset += self.problem.n_theta_opt

        assert v_offset == decision_variables.numel()

        return x, y, u, eta, p_opt, theta_opt

    def discretize(self, x_0=None, p=None, theta=None, last_u=None):
        """Discretize the OCP, returning a Optimization Problem

        :param x_0: initial condition
        :param p: parameters
        :param theta: theta parameters
        :param last_u: last applied control
        :rtype: NonlinearOptimizationProblem
        """
        if p is None:
            p = []
        if theta is None:
            theta = dict([(i, []) for i in range(self.finite_elements)])
        if x_0 is None:
            x_0 = self.problem.x_0

        # Create nlp object
        nlp = NonlinearOptimizationProblem(name='collocation_' + self.problem.name)

        # Create NLP symbolic variables
        all_decision_vars, x_var, y_var, u_var, eta, p_opt, theta_opt = self.create_nlp_symbolic_variables(nlp)
        cost = 0

        # Put the symbolic optimization parameters in the parameter vector
        for i, p_opt_index in enumerate(self.problem.get_p_opt_indices()):
            p[p_opt_index] = p_opt[i]

        # Put the symbolic theta_opt in the theta vector
        for i, theta_opt_index in enumerate(self.problem.get_theta_opt_indices()):
            for el in range(self.finite_elements):
                theta[el][theta_opt_index] = theta_opt[el][i]

        # Create "simulations" time_dict, a dict informing the simulation points in each finite elem. for each variable
        time_dict = self._create_time_dict_for_multiple_shooting()

        # Initial time constraint/initial condition
        f_h_initial = Function('h_initial', [self.model.x_sym, self.model.x_0_sym], [self.problem.h_initial])
        nlp.include_equality(f_h_initial(x_var[0][0], x_0))

        # Create functions to be evaluated
        functions = defaultdict(dict)
        for i in range(self.problem.n_g_ineq):
            functions['g_ineq_' + str(i)] = self._create_function_from_expression('f_g_ineq_' + str(i),
                                                                                  self.problem.g_ineq[i])

        for i in range(self.problem.n_g_eq):
            functions['g_eq_' + str(i)] = self._create_function_from_expression('f_g_eq_' + str(i),
                                                                                self.problem.g_eq[i])

        functions['f_s_cost'] = self._create_function_from_expression('f_s_cost',
                                                                      self.problem.S)

        # Multiple Shooting "simulation"
        results = self.get_system_at_given_times(x_var, y_var, u_var, time_dict, p, theta, functions=functions)

        # Build the NLP
        s_cost = 0
        for el in range(self.finite_elements):
            x_at_el_p_1 = results[el]['x'][0]
            y_end_el = results[el]['y'][0]

            # cost function as a sum of cost at each element
            if self.solution_method.solution_class == 'direct' and self.solution_method.cost_as_a_sum:
                x_at_el_p_1 = [x_at_el_p_1[n] for n in range(self.model.n_x)]
                x_at_el_p_1[-1] = 0
                x_at_el_p_1 = vertcat(*x_at_el_p_1)
                cost += results[el]['x'][0][-1]

            # Continuity constraint for defining x
            nlp.include_equality(x_at_el_p_1 - x_var[el + 1][0])

            # Defining y
            nlp.include_equality(y_end_el - y_var[el][0])

            # S cost
            s_cost += results[el]['f_s_cost'][-1]

            # Implement the constraint on delta_u
            if self.problem.has_delta_u:
                if el > 0:
                    for i in range(self.model.n_u):
                        if not is_equal(self.problem.delta_u_max[i], inf) or not is_equal(self.problem.delta_u_min[i],
                                                                                          -inf):
                            nlp.include_inequality(u_var[el][0][i] - u_var[el - 1][0][i],
                                                   lb=self.problem.delta_u_min[i], ub=self.problem.delta_u_max[i])

                elif el == 0 and last_u is not None:
                    for i in range(self.model.n_u):
                        if (not is_equal(self.problem.delta_u_max[i], inf)
                                or not is_equal(self.problem.delta_u_min[i], -inf)):
                            nlp.include_inequality(u_var[el][0][i] - last_u[i],
                                                   lb=self.problem.delta_u_min[i], ub=self.problem.delta_u_max[i])

            # Time dependent inequalities
            for i in range(self.problem.n_g_ineq):
                nlp.include_inequality(results[el]['g_ineq_' + str(i)][0], ub=0)

            # Time dependent equalities
            for i in range(self.problem.n_g_eq):
                nlp.include_equality(results[el]['g_eq_' + str(i)][0])

        # Final time constraint
        f_h_final = Function('h_final', [self.model.x_sym, self.problem.eta, self.model.p_sym], [self.problem.h_final])
        nlp.include_equality(f_h_final(x_var[-1][0], eta, p))

        # Time independent constraints
        f_h = Function('h', [self.model.p_sym], [self.problem.h])
        nlp.include_equality(f_h(p))

        if self.solution_method.solution_class == 'direct':
            if not self.solution_method.cost_as_a_sum:
                f_final_cost = Function('FinalCost', [self.model.x_sym, self.model.p_sym], [self.problem.V])
                cost = f_final_cost(x_var[-1][0], p)
            cost += s_cost
        nlp.set_objective(cost)

        return nlp

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
            dae_sys.convert_from_tau_to_time(self.time_breakpoints[el],
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

            for t_ind, t in enumerate(element_breakpoints):
                t_next = t
                p_i = vertcat(p, theta[el], self.vectorize(u_var[el]))

                # Do the simulation
                sim_result = dae_sys.simulate(x_init, t_0=t_init, t_f=t_next, p=p_i, y_0=self.problem.y_guess,
                                              integrator_type=self.solution_method.integrator_type,
                                              integrator_options={'name': 'integrator_' + str(el) + '_' + str(t_ind)}
                                              )
                # Fetch values from results
                x_t, yz_t = sim_result['xf'], sim_result['zf']
                y_t, z_t = self.model.slice_yz_to_y_and_z(yz_t)

                # Save to the result vector
                if 'x' in time_dict[el] and t in time_dict[el]['x']:
                    results[el]['x'].append(x_t)
                if 'y' in time_dict[el] and t in time_dict[el]['y']:
                    results[el]['y'].append(yz_t)
                if 'u' in time_dict[el] and t in time_dict[el]['u']:
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

    def _create_time_dict_for_multiple_shooting(self):
        time_dict = defaultdict(dict)
        for el in range(self.finite_elements):
            time_dict[el]['t_0'] = self.time_breakpoints[el]
            time_dict[el]['t_f'] = self.time_breakpoints[el + 1]
            time_dict[el]['x'] = [self.time_breakpoints[el + 1]]
            time_dict[el]['y'] = [self.time_breakpoints[el + 1]]

            time_dict[el]['f_s_cost'] = [self.time_breakpoints[el + 1]]

            for i in range(self.problem.n_g_ineq):
                if self.problem.time_g_ineq[i] == 'start':
                    time_dict[el]['g_ineq_' + str(i)] = [self.time_breakpoints[el]]
                if self.problem.time_g_ineq[i] == 'end':
                    time_dict[el]['g_ineq_' + str(i)] = [self.time_breakpoints[el + 1]]
                if self.problem.time_g_ineq[i] == 'default':
                    time_dict[el]['g_ineq_' + str(i)] = [self.time_breakpoints[el + 1]]

            for i in range(self.problem.n_g_eq):
                if self.problem.time_g_eq[i] == 'start':
                    time_dict[el]['g_eq_' + str(i)] = [self.time_breakpoints[el]]
                if self.problem.time_g_eq[i] == 'end':
                    time_dict[el]['g_eq_' + str(i)] = [self.time_breakpoints[el + 1]]
                if self.problem.time_g_eq[i] == 'default':
                    time_dict[el]['g_eq_' + str(i)] = [self.time_breakpoints[el + 1]]

        return time_dict

    def create_initial_guess(self, p=None, theta=None):
        """Create an initial guess for the optimal control problem using problem.x_0, problem.y_guess, problem.u_guess,
        and a given p and theta (for p_opt and theta_opt) if they are given.
        If y_guess or u_guess are None the initial guess uses a vector of zeros of appropriate size.

        :param p: Optimization parameters
        :param theta: Optimization theta
        :return:
        """
        x_init = repmat(self.problem.x_0, self.finite_elements + 1)

        if self.problem.y_guess is not None:
            y_init = repmat(self.problem.y_guess, self.degree * self.finite_elements)
        else:
            y_init = repmat(DM.zeros(self.model.n_y), self.degree * self.finite_elements)

        if self.problem.u_guess is not None:
            u_init = repmat(self.problem.u_guess, self.degree_control * self.finite_elements)
        else:
            u_init = repmat(DM.zeros(self.model.n_u), self.degree_control * self.finite_elements)

        eta_init = DM.zeros(self.problem.n_eta, 1)
        p_opt_init = DM.zeros(self.problem.n_p_opt, 1)
        theta_opt_init = DM.zeros(self.problem.n_theta_opt * self.finite_elements, 1)

        if p is not None:
            for k, ind in enumerate(self.problem.get_p_opt_indices()):
                p_opt_init[k] = p[ind]

        if theta is not None:
            for el in range(self.finite_elements):
                for k, ind in enumerate(self.problem.get_theta_opt_indices()):
                    theta_opt_init[k + el * self.problem.n_theta_opt] = theta[el][ind]

        return vertcat(x_init, y_init, u_init, eta_init, p_opt_init, theta_opt_init)

    def create_initial_guess_with_simulation(self, u=None, p=None, theta=None):
        """Create an initial guess for the optimal control problem using by simulating with a given control u,
        and a given p and theta (for p_opt and theta_opt) if they are given.
        If no u is given the value of problem.u_guess is used, or problem.last_u, then a vector of zeros of appropriate
        size is used.
        If no p or theta is given, an vector of zeros o appropriate size is used.

        :param u: Control initial guess
        :param p: Optimization parameters
        :param theta: Optimization theta
        :return:
        """
        x_init = []
        y_init = []
        u_init = []

        # Simulation

        if u is None:
            if self.problem.u_guess is not None:
                u = self.problem.u_guess
            elif self.problem.last_u is not None:
                u = self.problem.last_u
            else:
                u = DM.zeros(self.model.n_u)

        x_0 = self.problem.x_0
        x_init.append([x_0])
        for el in range(self.finite_elements):
            simulation_results = self.model.simulate(x_0, t_f=self.time_breakpoints[el + 1],
                                                     t_0=self.time_breakpoints[el],
                                                     u=u, p=p, theta=theta, y_0=self.problem.y_guess)

            x_init.append([simulation_results.x[-1]])
            y_init.append([simulation_results.y[-1]])
            u_init.append(simulation_results.u[:self.degree_control])
            x_0, _, _ = simulation_results.final_condition()

        x_init = self.vectorize(x_init)
        y_init = self.vectorize(y_init)
        u_init = self.vectorize(u_init)

        # Other variables

        eta_init = DM.zeros(self.problem.n_eta, 1)
        p_opt_init = DM.zeros(self.problem.n_p_opt, 1)
        theta_opt_init = DM.zeros(self.problem.n_theta_opt * self.finite_elements, 1)

        if p is not None:
            for k, ind in enumerate(self.problem.get_p_opt_indices()):
                p_opt_init[k] = p[ind]

        if theta is not None:
            for el in range(self.finite_elements):
                for k, ind in enumerate(self.problem.get_theta_opt_indices()):
                    theta_opt_init[k + el * self.problem.n_theta_opt] = theta[el][ind]

        return vertcat(x_init, y_init, u_init, eta_init, p_opt_init, theta_opt_init)

    def set_data_to_optimization_result_from_raw_data(self, optimization_result, raw_solution_dict):
        """
        Set the raw data received from the solver and put it in the Optimization Result object
        :param optimization_result: OptimizationResult
        :param raw_solution_dict: dict
        """

        optimization_result.raw_solution_dict = raw_solution_dict
        optimization_result.raw_decision_variables = raw_solution_dict['x']

        optimization_result.objective = raw_solution_dict['f']
        optimization_result.constraint_values = raw_solution_dict['g']

        x_values, y_values, u_values, eta, p_opt, theta_opt = self.unpack_decision_variables(raw_solution_dict['x'])

        optimization_result.p_opt = p_opt
        optimization_result.eta = eta
        optimization_result.p_opt = p_opt
        optimization_result.theta_opt = theta_opt

        optimization_result.x_interpolation_data['values'] = x_values
        optimization_result.y_interpolation_data['values'] = y_values
        optimization_result.u_interpolation_data['values'] = u_values

        optimization_result.x_interpolation_data['time'] = [[t] for t in self.time_breakpoints]
        optimization_result.y_interpolation_data['time'] = [[t] for t in self.time_breakpoints[1:]]

        if self.degree_control == 1:
            optimization_result.u_interpolation_data['time'] = [[t] for t in self.time_breakpoints[:-1]]
        else:
            optimization_result.u_interpolation_data['time'] = [
                [t + self.delta_t * col for col in self.solution_method.collocation_points(self.degree_control)] for t
                in
                self.time_breakpoints[:-1]]
