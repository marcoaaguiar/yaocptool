# -*- coding: utf-8 -*-
"""
Created on

@author: Marco Aurelio Schmitz de Aguiar
"""
import itertools
import logging
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Dict, List, Literal, Optional, Tuple, Type, Union, cast

import matplotlib.pyplot as plt
import networkx.algorithms.coloring
from casadi import DM, SX, inf, inv, vertcat
from erised import Proxy

from yaocptool import create_polynomial_approximation, find_variables_indices_in_vector
from yaocptool.methods import AugmentedLagrangian, SolutionMethodInterface
from yaocptool.methods.augmented_lagrangian import (
    ALExogeunousData,
    AugmentedLagrangianOptions,
)
from yaocptool.methods.base.distributed_optimization_result import (
    DistibutedOptimizationResult,
)
from yaocptool.methods.base.optimizationresult import OptimizationResult
from yaocptool.methods.base.solutionmethodsbase import SolutionMethodsBase
from yaocptool.methods.network.intermediary_node_solution_method import (
    IntermediaryNodeSolutionMethod,
)
from yaocptool.modelling import Network
from yaocptool.modelling.network.node import Node
from yaocptool.util.util import Timer

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


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
    block_choice_method: Union[
        Literal["cyclic"], Literal["sample"], Literal["choices"]
    ] = "cyclic"
    debug_skip_replace_by_approximation: bool = False

    al_options: AugmentedLagrangianOptions = AugmentedLagrangianOptions()

    def __post_init__(self):
        if not isinstance(self.al_options, AugmentedLagrangianOptions):
            self.al_options = AugmentedLagrangianOptions(**self.al_options)


class DistributedAugmentedLagrangian(SolutionMethodInterface):
    def __init__(
        self,
        network: Network,
        solution_method_class: Type[SolutionMethodsBase],
        solution_method_options: dict = None,
        blocks: Dict[int, List[Node]] = None,
        options={},
    ):
        if solution_method_options is None:
            solution_method_options = {}

        self.network = network
        self.solution_method_class = solution_method_class
        self.solution_method_options = solution_method_options

        self.options = DistributedAugmentedLagrangianOptions(**options)

        if blocks is not None:
            self.blocks = blocks
        else:
            coloring: Dict[Node, int] = networkx.algorithms.coloring.greedy_color(
                self.network.graph
            )
            self.blocks = {
                group_number: [node for node, _ in group]
                for group_number, group in itertools.groupby(
                    coloring.items(), key=lambda item: item[1]
                )
            }
        self.nodes_proxies: Dict[Node, Proxy] = {}

        self.relax_dict = {}
        self._last_result: Dict[Node, OptimizationResult] = {}

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

    def _create_variable_approx(self, var: SX, tau: SX) -> Tuple[SX, SX]:
        approx_name = [var[i].name() + "_appr" for i in range(var.numel())]
        return create_polynomial_approximation(
            tau,
            var.shape[0],
            self.options.degree,
            name=approx_name,
        )

    def _get_variable_last_result(
        self,
        node: Node,
        variable_type: Union[Literal["y"], Literal["u"]],
        var_indices: Union[List[int], int],
    ) -> List[List[DM]]:
        if not isinstance(var_indices, list):
            var_indices = [var_indices]

        # if there is result
        if self._last_result.get(node) is not None:
            result = self._last_result[node]
            return result.get_variable(variable_type, var_indices)

        if variable_type == "y" and node.problem.y_guess is not None:
            return [
                [node.problem.y_guess[var_indices] for _ in range(self.options.degree)]
                for _ in range(self.options.finite_elements)
            ]

        if variable_type == "u" and node.problem.u_guess is not None:
            return [
                [node.problem.u_guess[var_indices] for _ in range(self.options.degree)]
                for _ in range(self.options.finite_elements)
            ]
        return [
            [DM.zeros(len(var_indices)) for _ in range(self.options.degree)]
            for _ in range(self.options.finite_elements)
        ]

    def _get_block_number(self, node: Node) -> int:
        for block_id, nodes in self.blocks.items():
            if node in nodes:
                return block_id
        raise ValueError("Node not found in any block")

    def _get_node_exogenous(self, node: Node) -> Dict[str, Dict[str, ALExogeunousData]]:
        data: Dict[str, Dict[str, ALExogeunousData]] = {"in": {}, "out": {}}
        for p in self.network.graph.pred[node]:
            block_number = self._get_block_number(p)
            connections = self.network.connections[p, node]
            y = connections["y"]  # external
            _ = connections["u"]  # model variable

            indices = find_variables_indices_in_vector(y, p.model.y)

            for ind in indices:
                data["in"][p.model.y[ind].name()] = {
                    "data": self._get_variable_last_result(p, "y", ind),
                    "block_number": block_number,
                }

        for s in self.network.graph.succ[node]:
            block_number = self._get_block_number(s)
            connections = self.network.connections[node, s]
            _ = connections["y"]  # model variable
            u = connections["u"]  # external

            indices = find_variables_indices_in_vector(u, s.model.u)
            for ind in indices:
                data["out"][s.model.u[ind].name()] = {
                    "data": self._get_variable_last_result(s, "u", ind),
                    "block_number": block_number,
                }

        return data

    def _get_node_theta(self, node: Node) -> Dict[int, DM]:
        data: Dict[int, DM] = defaultdict(DM)
        for p in self.network.graph.pred[node]:
            connections = self.network.graph.edges[p, node]
            y = connections["y"]
            u = connections["u"]
            print(p, y, node, u)

            indices = find_variables_indices_in_vector(y, p.model.y)
            for ind in indices:
                y_data = self._get_variable_last_result(p, "y", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *y_data[i])

        print(data)
        __import__("ipdb").set_trace()
        for s in self.network.graph.succ[node]:
            connections = self.network.graph.edges[node, s]
            y = connections["y"]
            u = connections["u"]
            print(node, y, s, u)

            indices = find_variables_indices_in_vector(u, s.model.u)
            for ind in indices:
                u_data = self._get_variable_last_result(s, "u", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *u_data[i])

                print(data)
                __import__("ipdb").set_trace()

        return dict(data)

    def _solve_node_problems(
        self, blocks: List[List[Node]]
    ) -> Dict[Node, OptimizationResult]:

        if self.options.distributed:
            return self._solve_node_problems_distributed(blocks)
        return self._solve_node_problems_serial(blocks)

    def _solve_node_problems_serial(
        self, blocks: List[List[Node]]
    ) -> Dict[Node, OptimizationResult]:

        result_dict = {}
        for node in itertools.chain(*blocks):
            # get node theta
            node_theta = self._get_node_theta(node)
            print(node_theta)
            # warm start
            initial_guess = (
                self._last_result[node].raw_solution_dict["x"]
                if node in self._last_result
                else None
            )

            # solve node problem
            #  if node.name.lower().startswith("dummy"):
            result = node.solution_method.solve(
                theta=node_theta, initial_guess=initial_guess
            )
            print(result)
            # FIXME: Undo
            #  result_dict[node] = self._last_result[node] = result
        return result_dict

    def _solve_node_problems_distributed(
        self, blocks: List[List[Node]]
    ) -> Dict[Node, OptimizationResult]:
        result_dict = {}
        node_theta = {}
        initial_guess = {}
        for block in blocks:
            for node in block:
                # get node theta
                node_theta[node] = self._get_node_theta(node)
                # warm start
                initial_guess[node] = (
                    self._last_result[node].raw_solution_dict["x"]
                    if node in self._last_result
                    else None
                )
            for node in block:
                if node not in self.nodes_proxies:
                    self.nodes_proxies[node] = Proxy(node)

            futures = {
                node: self.nodes_proxies[node].solution_method.solve(
                    theta=node_theta[node],
                    initial_guess=initial_guess[node],
                )
                for node in block
            }
            block_results = {node: future.result() for node, future in futures.items()}

            result_dict.update(block_results)
            self._last_result.update(block_results)
        return result_dict

    def _number_of_inner_iterations(self, outer_iteration: int) -> int:
        return (
            max(
                self.options.max_iter_inner,
                self.options.max_iter_inner_first,
            )
            if outer_iteration == 0 and self.options.max_iter_inner_first is not None
            else self.options.max_iter_inner
        )

    def _block_iterator(self):
        if self.options.block_choice_method == "cyclic":
            return [*self.blocks.values()]

        if self.options.block_choice_method == "choices":
            return random.choices([*self.blocks.values()], k=len(self.blocks))

        if self.options.block_choice_method == "sample":
            return random.sample([*self.blocks.values()], k=len(self.blocks))

        raise ValueError(
            "DistributedAugmentedLagrangianOptions.block_choice_method"
            f" is not valid: {self.options.block_choice_method}"
        )

    def solve(self) -> DistibutedOptimizationResult:
        self._include_exogenous_variables()
        self.prepare()

        # Create Augmented Lagrangian
        self.initialize()

        # initialize dicts
        start_time = time.time()
        result_dict = {}
        node_error = {}
        node_error_last: Dict[Node, DM] = {}
        max_error = inf
        outer_it = inner_it = -1  # no out of bounds

        total_block_iterations = 0
        total_inner_iterations = 0
        try:
            with Timer() as solve_timer:
                for outer_it in range(self.options.max_iter_outer):
                    LOGGER.info(
                        "Starting outer iteration: {}".format(outer_it).center(40, "=")
                    )
                    last_objective = DM.inf()
                    for inner_it in range(self._number_of_inner_iterations(outer_it)):
                        LOGGER.info("==> Solving inner iteration: {}".format(inner_it))

                        blocks_iterator = self._block_iterator()

                        # solve all blocks
                        result_dict.update(self._solve_node_problems(blocks_iterator))
                        total_block_iterations += len(self.blocks)
                        total_inner_iterations += 1

                        objective: DM = sum(
                            result.objective_opt_problem
                            for result in result_dict.values()
                        )  # type: ignore
                        LOGGER.info(
                            f"Objective: {objective} {str(len(result_dict))+'/'+ str(len(self.network.nodes)) if len(result_dict)< len(self.network.nodes) else ''}"
                        )

                        if (
                            objective
                            > (1 - self.options.inner_loop_tol) * last_objective
                        ):
                            LOGGER.info("Objective decrease satisfied")
                            break
                        last_objective = objective

                    # update nu
                    node_error = self._update_parameters(result_dict)

                    # Print outer loop status
                    self._print_nodes_errors(node_error, node_error_last, result_dict)
                    LOGGER.info(
                        f"Obective: {sum(result.objective_opt_problem for result in result_dict.values())}"
                    )
                    LOGGER.info(
                        "Ending outer iteration: {}".format(outer_it).center(40, "=")
                    )

                    # error
                    node_error_last = node_error.copy()
                    max_error = max(
                        *itertools.chain(*(val.nz for val in node_error.values()))
                    )
                    if max_error < self.options.abs_tol:
                        LOGGER.info(
                            "=== Exiting: {} | Viol. Error: {} | Total time: {} ===",
                            "Tolerance met",
                            max_error,
                            time.time() - start_time,
                        )
                        break
                else:
                    LOGGER.info(
                        "=== Exiting: {} | Iterations: {} | Viol. Error: {:e} | Total time: {} ===".format(
                            "Max iteration reached",
                            outer_it,
                            float(max_error),
                            time.time() - start_time,
                        )
                    )
        finally:
            for proxy in self.nodes_proxies.values():
                proxy.terminate()

        return DistibutedOptimizationResult(
            results=result_dict,
            objective=sum(
                result.objective_opt_problem for result in result_dict.values()
            ),
            errors=node_error,
            number_of_iterations={
                "inner": inner_it,
                "outer": outer_it,
                "block_iterations": total_block_iterations,
            },
            times={"solution": solve_timer.elapsed},
        )

    def _update_parameters(self, result_dict: Dict[Node, OptimizationResult]):
        if self.options.distributed:
            return self._update_parameters_distributed(result_dict)
        return self._update_parameters_serial(result_dict)

    def _compute_primal_dual_mu(self, mu: DM, r_k_list: List[DM], s_k_list: List[DM]):
        r_k_1 = max(*r_k_list)
        s_k_1 = max(*s_k_list)
        LOGGER.info(f"Primal: {r_k_1}, Dual: {s_k_1}")

        if r_k_1 > self.options.al_options.gamma * s_k_1:
            old_mu = mu
            mu = self.options.al_options.beta * mu
            LOGGER.info(f"Increasing mu {old_mu} -> {mu[0]}")
        elif s_k_1 > self.options.al_options.gamma * r_k_1:
            old_mu = mu
            mu = inv(self.options.al_options.beta_decrease) * mu
            LOGGER.info(f"Decreasing mu {old_mu} -> {mu[0]}")
        else:
            LOGGER.info(f"Same mu {mu[0]}")
        return mu

    def _update_parameters_distributed(
        self, result_dict: Dict[Node, OptimizationResult]
    ) -> Dict[Node, DM]:
        node_error_future = {}
        node_errors = {}

        if self.options.al_options.mu_update_rule == "primal-dual":
            mu_by_node = {
                node: self.nodes_proxies[node].solution_method.mu.retrieve().result()
                for node in self.network.nodes
            }
            mu_set = {float(mu[0]) for mu in mu_by_node.values()}
            r_s_by_node_futures = {
                node: self.nodes_proxies[node].solution_method.compute_primal_dual(
                    #  node: dict(
                    p=None,
                    theta=self._get_node_theta(node),
                    raw_solution_dict=result.raw_solution_dict,
                    exogenous_data=self._get_node_exogenous(node),
                )
                for node, result in result_dict.items()  # if not node.name.lower().startswith("dummy")
            }
            r_s_by_node = {
                node: future.result() for node, future in r_s_by_node_futures.items()
            }

            if len(mu_set) > 1:
                raise ValueError("all mu should have the same value")
            mu = DM(mu_set.pop())
            r_k_list = [r_s["primal"] for r_s in r_s_by_node.values()]
            s_k_list = [r_s["dual"] for r_s in r_s_by_node.values()]
            node_errors = {node: r_s["error"] for (node, r_s) in r_s_by_node.items()}

            mu = self._compute_primal_dual_mu(mu, r_k_list, s_k_list)
            for node in result_dict:
                self.nodes_proxies[node].solution_method.set_mu(mu).result()
        else:
            for node, result in result_dict.items():
                node_theta = self._get_node_theta(node)
                node_exog = self._get_node_exogenous(node)
                raw_solution_dict = result.raw_solution_dict

                node_error_future[node] = self.nodes_proxies[
                    node
                ].solution_method.update_parameters(
                    p=None,
                    theta=node_theta,
                    raw_solution_dict=raw_solution_dict,
                    exogenous_data=node_exog,
                )
                if self.options.al_options.mu_update_rule == "primal-dual":
                    mu_by_node = {
                        node: cast(
                            DM,
                            self.nodes_proxies[node]
                            .solution_method.mu.retrieve()
                            .result(),
                        )
                        for node in self.nodes_proxies
                    }
                    mu_set = {float(mu[0]) for mu in mu_by_node.values()}
                    r_s_by_node = {
                        node: cast(
                            Dict[str, DM],
                            self.nodes_proxies[node]
                            .solution_method.compute_primal_dual(
                                p=None,
                                theta=node_theta,
                                raw_solution_dict=raw_solution_dict,
                                exogenous_data=node_exog,
                            )
                            .result(),
                        )
                        for node in self.nodes_proxies
                        if not node.name.lower().startswith("dummy")
                    }

                    if len(mu_set) > 1:
                        raise ValueError("all mu should have the same value")
                    mu = mu_set.pop()
                    r_k_list = [r_s["primal"] for r_s in r_s_by_node.values()]
                    s_k_list = [r_s["dual"] for r_s in r_s_by_node.values()]

                    mu = self._compute_primal_dual_mu(mu, r_k_list, s_k_list)

                node_errors = {
                    node: future.result() for node, future in node_error_future.items()
                }

            # sync mu
            nodes_mu_max = {
                node: cast(
                    DM, self.nodes_proxies[node].solution_method.mu.retrieve().result()
                )
                for node in self.nodes_proxies
            }

            max_mu = max(max(v.nz) for v in nodes_mu_max.values())

            for node, mu in nodes_mu_max.items():
                self.nodes_proxies[node].solution_method.mu = max_mu * DM.ones(mu.shape)

            for node, mu in nodes_mu_max.items():
                self.nodes_proxies[node].wait_all_futures()

        return node_errors

    def _update_parameters_serial(
        self, result_dict: Dict[Node, OptimizationResult]
    ) -> Dict[Node, DM]:
        node_errors = {}

        if self.options.al_options.mu_update_rule == "primal-dual":
            mu_by_node = {node: node.solution_method.mu for node in self.network.nodes}
            mu_set = {float(mu[0]) for mu in mu_by_node.values()}
            r_s_by_node = {
                node: node.solution_method.compute_primal_dual(
                    p=None,
                    theta=self._get_node_theta(node),
                    raw_solution_dict=result.raw_solution_dict,
                    exogenous_data=self._get_node_exogenous(node),
                )
                for node, result in result_dict.items()
            }

            if len(mu_set) > 1:
                raise ValueError("all mu should have the same value")
            mu = DM(mu_set.pop())
            r_k_list = [r_s["primal"] for r_s in r_s_by_node.values()]
            s_k_list = [r_s["dual"] for r_s in r_s_by_node.values()]
            node_errors = {node: r_s["error"] for (node, r_s) in r_s_by_node.items()}

            mu = self._compute_primal_dual_mu(mu, r_k_list, s_k_list)
            for node in result_dict:
                node.solution_method.set_mu(mu)
        else:
            for node, result in sorted(
                result_dict.items(), key=(lambda item: item[0].node_id)
            ):
                node_theta = self._get_node_theta(node)
                node_exog = self._get_node_exogenous(node)
                raw_solution_dict = result.raw_solution_dict

                node_errors[node] = node.solution_method.update_parameters(
                    p=None,
                    theta=node_theta,
                    raw_solution_dict=raw_solution_dict,
                    exogenous_data=node_exog,
                )  # type: ignore
        return node_errors

    def _create_solution_methods(self):
        for block, nodes in self.blocks.items():
            for node in nodes:
                if node.name.startswith("Dummy") or node.name.startswith("dummy"):
                    node.solution_method = IntermediaryNodeSolutionMethod(
                        node.problem,
                        self.solution_method_class,
                        solver_options=self.solution_method_options,
                        relax_algebraic_index=self.relax_dict[node]["alg_relax_ind"],
                        relax_algebraic_var_index=self.relax_dict[node]["y_relax"],
                        relax_time_equality_index=self.relax_dict[node]["eq_relax_ind"],
                        no_update_after_solving=True,
                        block_number=block,
                        options={
                            **asdict(self.options.al_options),
                            **{
                                "degree": self.options.degree,
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
                        block_number=block,
                        options={
                            **asdict(self.options.al_options),
                            **{
                                "degree": self.options.degree,
                                "finite_elements": self.options.finite_elements,
                                "max_iter": 1,
                                "verbose": 0,
                            },
                        },
                    )
                    node.solution_method.create_optimization_problem()

    def _print_nodes_errors(
        self,
        node_error: Dict[Node, DM],
        node_error_last: Dict[Node, DM],
        result_dict: Dict[Node, OptimizationResult],
    ):
        max_name_len = max(len(n.name) for n in self.network.nodes)
        for node in sorted(result_dict, key=lambda x: x.node_id):
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

            LOGGER.info(
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

    def initialize(self):
        self._create_solution_methods()

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
