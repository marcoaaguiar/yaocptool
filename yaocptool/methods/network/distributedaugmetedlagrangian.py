# -*- coding: utf-8 -*-
"""
Created on 

@author: Marco Aurelio Schmitz de Aguiar
"""
import time
from collections import defaultdict

import matplotlib.pyplot as plt
from casadi import inf

from yaocptool import create_polynomial_approximation, find_variables_indices_in_vector, DM, vertcat, join_thetas
from yaocptool.methods import SolutionMethodInterface, AugmentedLagrangian


class DistributedAugmentedLagrangian(SolutionMethodInterface):
    def __init__(self,
                 network,
                 solution_method_class,
                 solution_method_options=None,
                 **kwargs):
        """

        :param yaocptool.modelling.Network network:
        :param type solution_method_class:
        :param dict solution_method_options:
        """
        if solution_method_options is None:
            solution_method_options = {}

        self.network = network
        self.solution_method_class = solution_method_class
        self.solution_method_options = solution_method_options

        self.degree = 3
        self.finite_elements = 20
        self.max_iter_inner = 4
        self.max_iter_inner_first = 4
        self.max_iter_outer = 30
        self.abs_tol = 1e-6
        self.mu_0 = 1.0
        self.mu_max = 1e6
        self.relax_dict = {}
        self._last_result = {}  # dict

        self._debug_skip_replace_by_approximation = False
        self._debug_skip_update_nu = False
        self._debug_skip_update_mu = False

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def _include_exogenous_variables(self):
        for node in self.network.nodes:
            if node not in self.relax_dict:
                self.relax_dict[node] = {
                    'u': [],
                    'y_relax': [],
                    'alg_relax_ind': [],
                    'eq_relax_ind': []
                }

            for p in self.network.graph.pred[node]:
                y = self.network.graph.edges[p, node]['y']
                u = self.network.graph.edges[p, node]['u']

                # create an approximation
                y_appr_name = [y[i].name() + '_appr' for i in range(y.numel())]
                y_appr, y_appr_par = create_polynomial_approximation(
                    node.problem.model.tau,
                    y.shape[0],
                    self.degree,
                    name=y_appr_name)

                # register the added connection to the list of algebraic equations to be relaxed
                self.relax_dict[node]['alg_relax_ind'].extend(
                    list(
                        range(node.problem.model.alg.numel(),
                              node.problem.model.alg.numel() + u.numel())))
                if not self._debug_skip_replace_by_approximation:
                    node.problem.connect(u, y_appr)
                else:
                    node.problem.connect(u, y)

                node.problem.include_theta(y_appr_par)
                self.relax_dict[node]['y_relax'].extend(
                    find_variables_indices_in_vector(u,
                                                     node.problem.model.y_sym))

            for s in self.network.graph.succ[node]:
                y = self.network.graph.edges[node, s]['y']
                u = self.network.graph.edges[node, s]['u']

                u_appr_name = [u[i].name() + '_appr' for i in range(u.numel())]
                u_appr, u_appr_par = create_polynomial_approximation(
                    node.problem.model.tau,
                    u.shape[0],
                    self.degree,
                    name=u_appr_name)

                self.relax_dict[node]['eq_relax_ind'].extend(
                    list(
                        range(node.problem.g_eq.numel(),
                              node.problem.g_eq.numel() + y.numel())))
                if not self._debug_skip_replace_by_approximation:
                    node.problem.include_time_equality(y - u_appr)
                else:
                    node.problem.include_time_equality(y - u)

                node.problem.include_theta(u_appr_par)

    def _get_variable_last_result(self, node, variable_type, var_indices):
        """

        :param yaocptool.modelling.Node node:
        :param str variable_type:
        :param int|list of int var_indices:
        :return:
        """
        if not isinstance(var_indices, list):
            var_indices = [var_indices]

        # If no result is available
        if self._last_result.get(node) is None:
            if variable_type == 'y' and node.problem.y_guess is not None:
                return [[
                    node.problem.y_guess[var_indices]
                    for _ in range(self.degree)
                ] for i in range(self.finite_elements)]
            elif variable_type == 'u' and node.problem.u_guess is not None:
                return [[
                    node.problem.u_guess[var_indices]
                    for j in range(self.degree)
                ] for i in range(self.finite_elements)]
            return [[DM.zeros(len(var_indices)) for j in range(self.degree)]
                    for i in range(self.finite_elements)]

        result = self._last_result[node]
        return result.get_variable(variable_type, var_indices)

    def _get_node_theta(self, node):
        data = defaultdict(list)
        for p in self.network.graph.pred[node]:
            y = self.network.graph.edges[p, node]['y']
            u = self.network.graph.edges[p, node]['u']

            indices = find_variables_indices_in_vector(y, p.model.y)
            for ind in indices:
                y_data = self._get_variable_last_result(p, 'y', ind)
                for i in range(self.finite_elements):
                    data[i] = vertcat(data[i], *y_data[i])

        for s in self.network.graph.succ[node]:
            y = self.network.graph.edges[node, s]['y']
            u = self.network.graph.edges[node, s]['u']

            indices = find_variables_indices_in_vector(u, s.model.u)
            for ind in indices:
                u_data = self._get_variable_last_result(s, 'u', ind)
                for i in range(self.finite_elements):
                    data[i] = vertcat(data[i], *u_data[i])

        return data

    def solve(self):
        self.prepare()

        self._include_exogenous_variables()

        # Create Augmented Lagrangian
        for node in self.network.nodes:
            node.solution_method = AugmentedLagrangian(
                node.problem,
                self.solution_method_class,
                solver_options=self.solution_method_options,
                relax_algebraic_index=self.relax_dict[node]['alg_relax_ind'],
                relax_algebraic_var_index=self.relax_dict[node]['y_relax'],
                relax_time_equality_index=self.relax_dict[node]
                ['eq_relax_ind'],
                degree=self.degree,
                degree_control=self.degree,
                finite_elements=self.finite_elements,
                no_update_after_solving=True,
                max_iter=1,
                mu_0=self.mu_0,
                mu_max=self.mu_max,
                # _debug_skip_update_nu=True,
                # _debug_skip_update_mu=True,
                verbose=0)

        # initialize dicts
        node_theta = {}
        result_dict = {}
        node_error = {}
        node_error_last = dict([(node, 0.) for node in self.network.nodes])
        t_0 = time.time()
        max_error = inf
        for outer_it in range(self.max_iter_outer):
            print("Starting outer iteration: {}".format(outer_it).center(
                40, '='))
            for inner_it in range(
                    self.max_iter_inner_first if outer_it == 0 and self.
                    max_iter_inner_first is not None else self.max_iter_inner):
                print("Starting inner iteration: {}".format(inner_it).center(
                    30, '='))

                # for node in self.network.nodes:  # sorted(self.network.nodes, key=lambda x: x.node_id):
                for node in sorted(self.network.nodes,
                                   key=lambda x: x.node_id):
                    # print("Solving: {}".format(node))

                    # get node theta
                    node_theta[node] = self._get_node_theta(node)

                    # warm start
                    if self._last_result.get(node) is not None:
                        initial_guess_dict = self._last_result.get(
                            node).raw_solution_dict
                        initial_guess = initial_guess_dict['x']
                    else:
                        initial_guess_dict = None
                        initial_guess = None

                    # solve node problem
                    result = node.solution_method.solve(
                        theta=node_theta[node], initial_guess=initial_guess)

                    # save result
                    result_dict[node] = result
                    self._last_result[node] = result
                    # print("Finished: {}".format(node))

            # update nu
            for node in sorted(self.network.nodes, key=lambda x: x.node_id):
                node_theta[node] = self._get_node_theta(node)
                theta_k = join_thetas(node_theta[node],
                                      node.solution_method.nu)
                p_k = node.solution_method.mu
                raw_solution_dict = self._last_result[node].raw_solution_dict
                node_error[
                    node] = node.solution_method._compute_new_nu_and_error(
                        theta=theta_k,
                        p=p_k,
                        raw_solution_dict=raw_solution_dict)
                print("Violation node: {} | error: {} | diff: {}".format(
                    node.name, node_error[node],
                    node_error[node] - node_error_last[node]))
                node_error_last[node] = node_error[node]

            # update mu
            if not self._debug_skip_update_mu:
                for node in self.network.nodes:
                    node.solution_method._update_mu()

            print("Ending outer iteration: {}".format(outer_it).center(
                40, '='))

            # error
            max_error = max(node_error.values())
            if max_error < self.abs_tol:
                print('=== Exiting: {} | Viol. Error: {} | Total time: {} ==='.
                      format('Tolerance met', max_error,
                             time.time() - t_0))
                break
        else:
            print(
                '=== Exiting: {} | Iterations: {} | Viol. Error: {:e} | Total time: {} ==='
                .format('Max iteration reached', outer_it, float(max_error),
                        time.time() - t_0))

        return result_dict

    def prepare(self):
        for node in self.network.nodes:
            node.problem.pre_solve_check()

    def plot_all_relaxations(self, result_list):
        for k, edge in enumerate(self.network.graph.edges):
            print(k)
            fig = plt.figure(k)
            y_index = find_variables_indices_in_vector(
                self.network.graph.edges[edge]['y'], edge[0].problem.model.y)
            u_index = find_variables_indices_in_vector(
                self.network.graph.edges[edge]['u'], edge[1].problem.model.u)
            result_list[edge[0]].plot({'y': y_index},
                                      figures=[fig],
                                      show=False)
            result_list[edge[1]].plot({'u': u_index}, figures=[fig])
