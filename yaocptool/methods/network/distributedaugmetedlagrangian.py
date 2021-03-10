# -*- coding: utf-8 -*-
"""
Created on 

@author: Marco Aurelio Schmitz de Aguiar
"""
import time
import random
import concurrent.futures
from collections import defaultdict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Type
from yaocptool.modelling.network.node import Node

import matplotlib.pyplot as plt
from casadi import DM, SX, inf, vertcat
import networkx.algorithms.coloring

from yaocptool import (
    create_polynomial_approximation,
    find_variables_indices_in_vector,
    join_thetas,
)
from yaocptool.methods import AugmentedLagrangian, SolutionMethodInterface
from yaocptool.methods.augmented_lagrangian import AugmentedLagrangianOptions
from yaocptool.methods.network.intermediary_node_solution_method import (
    IntermediaryNodeSolutionMethod,
)
from yaocptool.modelling import Network
import itertools


@dataclass
class DistributedAugmentedLagrangianOptions:
    degree: int = 3
    finite_elements: int = 20
    max_iter_inner: int = 4
    max_iter_inner_first: Optional[int] = None
    max_iter_outer: int = 30
    abs_tol: float = 1e-6
    inner_loop_tol: float = 1e-4
    randomized_inner: bool = False
    distributed: bool = False

    debug_skip_replace_by_approximation: bool = False

    al_options: AugmentedLagrangianOptions = AugmentedLagrangianOptions()

    def __post_init__(self):
        if not isinstance(self.al_options, AugmentedLagrangianOptions):
            self.al_options = AugmentedLagrangianOptions(**self.al_options)


class DistributedAugmentedLagrangian(SolutionMethodInterface):
    def __init__(
        self,
        network: Network,
        solution_method_class: Type[SolutionMethodInterface],
        solution_method_options: dict = None,
        options={},
    ):
        if solution_method_options is None:
            solution_method_options = {}

        self.network = network
        self.solution_method_class = solution_method_class
        self.solution_method_options = solution_method_options

        self.options = DistributedAugmentedLagrangianOptions(**options)

        coloring: Dict[Node, int] = networkx.algorithms.coloring.greedy_color(
            self.network.graph
        )
        self.blocks = {
            group_number: [node for node, _ in group]
            for group_number, group in itertools.groupby(
                coloring.items(), key=lambda item: item[1]
            )
        }

        self.relax_dict = {}
        self._last_result = {}

    def _include_exogenous_variables(self):
        self.relax_dict = {
            node: {
                "u": [],
                "y_relax": [],
                "alg_relax_ind": [],
                "eq_relax_ind": [],
            }
            for node in self.network
        }

        for node in self.network:
            for p in self.network.graph.pred[node]:
                # bind `node` inputs (u)
                connection = self.network.graph.edges[p, node]

                y = connection["y"]
                u = connection["u"]

                # create an approximation
                y_appr, y_appr_par = self._create_variable_approx(
                    y, node.problem.model.tau
                )

                # register the added connection to the list of algebraic equations to be relaxed
                self.relax_dict[node]["alg_relax_ind"].extend(
                    range(
                        node.problem.model.alg.numel(),
                        node.problem.model.alg.numel() + u.numel(),
                    )
                )
                if not self.options.debug_skip_replace_by_approximation:
                    node.problem.connect(u, y_appr)  # u - y
                else:
                    node.problem.connect(u, y)

                node.problem.include_theta(y_appr_par)
                self.relax_dict[node]["y_relax"].extend(
                    find_variables_indices_in_vector(u, node.problem.model.y)
                )

            for s in self.network.graph.succ[node]:
                # bind `node` output (y)
                connection = self.network.graph.edges[node, s]
                y = connection["y"]
                u = connection["u"]

                u_appr, u_appr_par = self._create_variable_approx(
                    u, node.problem.model.tau
                )

                self.relax_dict[node]["eq_relax_ind"].extend(
                    range(
                        node.problem.g_eq.numel(),
                        node.problem.g_eq.numel() + y.numel(),
                    )
                )
                eq = (
                    u_appr - y
                    if not self.options.debug_skip_replace_by_approximation
                    else u - y
                )

                node.problem.include_time_equality(eq)
                node.problem.include_theta(u_appr_par)

    def _create_variable_approx(self, var: SX, tau: SX):
        approx_name = [var[i].name() + "_appr" for i in range(var.numel())]
        return create_polynomial_approximation(
            tau,
            var.shape[0],
            self.options.degree,
            name=approx_name,
        )

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
            if variable_type == "y" and node.problem.y_guess is not None:
                return [
                    [
                        node.problem.y_guess[var_indices]
                        for _ in range(self.options.degree)
                    ]
                    for i in range(self.options.finite_elements)
                ]
            elif variable_type == "u" and node.problem.u_guess is not None:
                return [
                    [
                        node.problem.u_guess[var_indices]
                        for j in range(self.options.degree)
                    ]
                    for i in range(self.options.finite_elements)
                ]
            return [
                [DM.zeros(len(var_indices)) for j in range(self.options.degree)]
                for i in range(self.options.finite_elements)
            ]

        result = self._last_result[node]
        return result.get_variable(variable_type, var_indices)

    def _get_node_theta(self, node: Node):
        data = defaultdict(list)
        for p in self.network.graph.pred[node]:
            connections = self.network.graph.edges[p, node]
            y = connections["y"]
            u = connections["u"]

            indices = find_variables_indices_in_vector(y, p.model.y)
            for ind in indices:
                y_data = self._get_variable_last_result(p, "y", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *y_data[i])

        for s in self.network.graph.succ[node]:
            connections = self.network.graph.edges[node, s]
            y = connections["y"]
            u = connections["u"]

            indices = find_variables_indices_in_vector(u, s.model.u)
            for ind in indices:
                u_data = self._get_variable_last_result(s, "u", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *u_data[i])

        return data

    def _solve_node_problems(self, nodes):
        if self.options.distributed:
            return self._solve_node_problems_distributed()
        return self._solve_node_problems_serial(nodes)

    def _solve_node_problems_serial(self, nodes):
        result_dict = {}
        for node in itertools.chain(nodes):
            # get node theta
            node_theta = self._get_node_theta(node)
            # warm start
            initial_guess = (
                self._last_result[node].raw_solution_dict["x"]
                if node in self._last_result
                else None
            )

            # solve node problem
            result = node.solution_method.solve(
                theta=node_theta, initial_guess=initial_guess
            )
            if "return_status" in result.stats and result.stats[
                "return_status"
            ] not in {
                "Search_Direction_Becomes_Too_Small",
                "Solved_To_Acceptable_Level",
                "Solve_Succeeded",
            }:
                __import__("ipdb").set_trace()
            result_dict[node] = self._last_result[node] = result
        return result_dict

    def _solve_node_problems_distributed(self):
        result_dict = {}
        node_theta = {}
        initial_guess = {}
        for block in self.blocks.values():
            for node in block:
                # get node theta
                node_theta[node] = self._get_node_theta(node)
                # warm start
                initial_guess[node] = (
                    self._last_result[node].raw_solution_dict["x"]
                    if node in self._last_result
                    else None
                )
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = {
                    node: executor.submit(
                        node.solution_method.solve,
                        theta=node_theta[node],
                        initial_guess=initial_guess[node],
                    )
                    for node in block
                }
                block_results = {
                    node: future.result() for node, future in futures.items()
                }

            for node in block:
                result = block_results[node]
                if "return_status" in result.stats and result.stats[
                    "return_status"
                ] not in {
                    "Search_Direction_Becomes_Too_Small",
                    "Solved_To_Acceptable_Level",
                    "Solve_Succeeded",
                }:
                    __import__("ipdb").set_trace()
                result_dict.update(block_results)
                self._last_result.update(block_results)
        return result_dict

    def solve(self):
        self._include_exogenous_variables()
        self.prepare()

        # Create Augmented Lagrangian
        self._create_solution_methods()

        # initialize dicts
        start_time = time.time()
        node_theta = {}
        result_dict = {}
        node_error = {}
        node_error_last = {}
        max_error = inf

        for outer_it in range(self.options.max_iter_outer):
            print("Starting outer iteration: {}".format(outer_it).center(40, "="))
            n_inner_iterations = (
                max(self.options.max_iter_inner, self.options.max_iter_inner_first)
                if outer_it == 0 and self.options.max_iter_inner_first is not None
                else self.options.max_iter_inner
            )
            last_objective = inf
            for inner_it in range(n_inner_iterations):
                print("==> Solving inner iteration: {}".format(inner_it))

                if self.options.randomized_inner:
                    nodes_iterator = random.sample(
                        self.network.nodes, len(self.network.nodes)
                    )
                else:
                    nodes_iterator = sorted(self.network.nodes, key=lambda x: x.node_id)

                result_dict = self._solve_node_problems(nodes_iterator)

                objective = sum(
                    result_dict[node].objective_opt_problem for node in self.network
                )
                print(f"Objective: {objective}")
                if objective > (1 - self.options.inner_loop_tol) * last_objective:
                    break
                last_objective = objective

            # update nu
            for node in sorted(self.network.nodes, key=lambda x: x.node_id):
                node_theta[node] = self._get_node_theta(node)
                theta_k = join_thetas(node_theta[node], node.solution_method.nu)
                p_k = node.solution_method.mu  # TODO: allow for models with parameters!
                raw_solution_dict = result_dict[node].raw_solution_dict

                node_error[node] = node.solution_method.update_parameters(
                    p=p_k, theta=theta_k, raw_solution_dict=raw_solution_dict
                )

            # Print outer loop status
            self._print_nodes_errors(node_error, node_error_last)
            print(
                f"Obective: {sum(result_dict[node].objective_opt_problem for node in self.network)}"
            )
            print("Ending outer iteration: {}".format(outer_it).center(40, "="))

            # error
            node_error_last = node_error.copy()
            max_error = max(*itertools.chain(*(val.nz for val in node_error.values())))
            if max_error < self.options.abs_tol:
                print(
                    "=== Exiting: {} | Viol. Error: {} | Total time: {} ===".format(
                        "Tolerance met", max_error, time.time() - start_time
                    )
                )
                break
        else:
            print(
                "=== Exiting: {} | Iterations: {} | Viol. Error: {:e} | Total time: {} ===".format(
                    "Max iteration reached",
                    outer_it,
                    float(max_error),
                    time.time() - start_time,
                )
            )

        return result_dict

    def _create_solution_methods(self):
        for node in self.network.nodes:
            if node.name.startswith("Dummy") or node.name.startswith("dummy"):
                node.solution_method = IntermediaryNodeSolutionMethod(
                    #  node.solution_method = AugmentedLagrangian(
                    node.problem,
                    self.solution_method_class,
                    solver_options=self.solution_method_options,
                    relax_algebraic_index=self.relax_dict[node]["alg_relax_ind"],
                    relax_algebraic_var_index=self.relax_dict[node]["y_relax"],
                    relax_time_equality_index=self.relax_dict[node]["eq_relax_ind"],
                    no_update_after_solving=True,
                    #  debug_skip_update_mu=False,
                    #  debug_skip_update_nu=False,
                    options={
                        **asdict(self.options.al_options),
                        **{
                            "degree": self.options.degree,
                            "degree_control": self.options.degree,
                            "finite_elements": self.options.finite_elements,
                            "max_iter": 1,
                            "verbose": 0,
                        },
                    },
                )
            else:
                node.solution_method = AugmentedLagrangian(
                    node.problem,
                    self.solution_method_class,
                    solver_options=self.solution_method_options,
                    relax_algebraic_index=self.relax_dict[node]["alg_relax_ind"],
                    relax_algebraic_var_index=self.relax_dict[node]["y_relax"],
                    relax_time_equality_index=self.relax_dict[node]["eq_relax_ind"],
                    no_update_after_solving=True,
                    #  debug_skip_update_mu=False,
                    #  debug_skip_update_nu=False,
                    options={
                        **asdict(self.options.al_options),
                        **{
                            "degree": self.options.degree,
                            "degree_control": self.options.degree,
                            "finite_elements": self.options.finite_elements,
                            "max_iter": 1,
                            "verbose": 0,
                        },
                    },
                )
                node.solution_method.create_optimization_problem()

    def _print_nodes_errors(self, node_error, node_error_last):
        max_name_len = max(len(n.name) for n in self.network.nodes)
        for node in sorted(self.network.nodes, key=lambda x: x.node_id):
            error_ = ", ".join(
                f"\033[94m{float(numb): .2e}\033[0m" for numb in node_error[node].nz
            )
            diff_color_ = lambda x: "\033[92m" if x < 0 else "\033[91m"
            if node in node_error_last:
                diff_ = ", ".join(
                    f"{diff_color_(numb)}{float(numb): .2e}"
                    for numb in (node_error[node] - node_error_last[node]).nz
                )
            else:
                diff_ = "-"

            print(
                "Node {} - \033[93m{:>{}}\033[0m | error: {} \033[0m| diff: {} \033[0m".format(
                    node.node_id,
                    node.name,
                    max_name_len,
                    error_,
                    diff_,
                )
            )

    def prepare(self):
        for node in self.network.nodes:
            node.problem.pre_solve_check()

    def plot_all_relaxations(self, result_list):
        for k, edge in enumerate(self.network.graph.edges):
            fig = plt.figure(k)
            y_index = find_variables_indices_in_vector(
                self.network.graph.edges[edge]["y"], edge[0].problem.model.y
            )
            u_index = find_variables_indices_in_vector(
                self.network.graph.edges[edge]["u"], edge[1].problem.model.u
            )
            result_list[edge[0]].plot({"y": y_index}, figures=[fig], show=False)
            result_list[edge[1]].plot({"u": u_index}, figures=[fig])
