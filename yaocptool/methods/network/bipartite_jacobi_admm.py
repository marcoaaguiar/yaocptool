from collections import defaultdict
from typing import Dict

from casadi import DM, vertcat

from yaocptool.methods.network.distributedaugmetedlagrangian import (
    DistributedAugmentedLagrangian,
)
from yaocptool.modelling.network.node import Node
from yaocptool.util.util import find_variables_indices_in_vector


class BipartiteJacobiADMM(DistributedAugmentedLagrangian):
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
                u_appr, u_appr_par = self._create_variable_approx(
                    u, node.problem.model.tau
                )

                # register the added connection to the list of algebraic equations to be relaxed
                self.relax_dict[node]["alg_relax_ind"].extend(
                    range(
                        node.problem.model.alg.numel(),
                        node.problem.model.alg.numel() + u.numel(),
                    )
                )
                node.problem.connect(u, (y_appr + u_appr) / 2.0)  # u - (u_k + y_k)/2

                node.problem.include_theta(y_appr_par)
                node.problem.include_theta(u_appr_par)

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
                y_appr, y_appr_par = self._create_variable_approx(
                    y, node.problem.model.tau
                )

                self.relax_dict[node]["eq_relax_ind"].extend(
                    range(
                        node.problem.g_eq.numel(),
                        node.problem.g_eq.numel() + y.numel(),
                    )
                )

                eq = (u_appr + y_appr) / 2.0 - y

                node.problem.include_time_equality(eq)
                node.problem.include_theta(u_appr_par)
                node.problem.include_theta(y_appr_par)

    def _get_node_theta(self, node: Node) -> Dict[int, DM]:
        data: Dict[int, DM] = defaultdict(DM)
        for p in self.network.graph.pred[node]:
            connections = self.network.graph.edges[p, node]
            y = connections["y"]
            u = connections["u"]

            y_indices = find_variables_indices_in_vector(y, p.model.y)
            u_indices = find_variables_indices_in_vector(u, node.model.u)
            for ind in y_indices:
                y_data = self._get_variable_last_result(p, "y", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *y_data[i])
            for ind in u_indices:
                u_data = self._get_variable_last_result(node, "u", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *u_data[i])

        for s in self.network.graph.succ[node]:
            connections = self.network.graph.edges[node, s]
            y = connections["y"]
            u = connections["u"]

            u_indices = find_variables_indices_in_vector(u, s.model.u)
            y_indices = find_variables_indices_in_vector(y, node.model.y)
            for ind in u_indices:
                u_data = self._get_variable_last_result(s, "u", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *u_data[i])
            for ind in y_indices:
                y_data = self._get_variable_last_result(node, "y", ind)
                for i in range(self.options.finite_elements):
                    data[i] = vertcat(data[i], *y_data[i])

        return dict(data)
