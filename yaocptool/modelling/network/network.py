# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:05:50 2017

@author: marco
"""
from typing import Any, List, Mapping, Optional, Tuple, TypedDict, Union

import matplotlib.pyplot as plt
import networkx
from casadi import DM, SX, vertcat
from networkx.classes.reportviews import NodeView

from yaocptool import find_variables_indices_in_vector
from yaocptool.modelling.network.node import Node
from yaocptool.modelling.ocp import OptimalControlProblem
from yaocptool.modelling.system_model import SystemModel


class ConnectionData(TypedDict):
    y: SX
    u: SX


class Network:
    def __init__(
        self,
        nodes: List[Node] = None,
        name: str = "network",
        intermediary_nodes: bool = False,
    ):
        """

        :param list of Node nodes:
        """

        self.name = name

        self.graph = networkx.DiGraph()  # type: networkx.DiGraph
        self._nodes_id_counter = 0

        self.intermediary_nodes = intermediary_nodes

        if nodes is not None:
            self.include_nodes(nodes)

    def __getitem__(self, key: Union[str, int]) -> Node:
        if isinstance(key, int):
            node = self.get_node_by_id(key)
            if node is None:
                raise IndexError(f"Network node index out of range: {key}")
            return node
        elif isinstance(key, str):
            node = self.get_node_by_name(key)
            if node is None:
                raise KeyError(f"{key}")
            return node
        raise NotImplementedError("Not implemeted for key with type: %s", type(key))

    @property
    def nodes(
        self,
    ) -> NodeView:
        """
        All nodes (from the self.graph)
        """
        return self.graph.nodes

    @property
    def connections(self) -> Mapping[Tuple[Node, Node], ConnectionData]:
        """
            All connections (edges) from the self.graph

        :rtype: networkx.reportviews.OutEdgeView
        """
        return self.graph.edges

    @property
    def models(self) -> List[SystemModel]:
        """
            List of models of all nodes

        :rtype: list of SystemModel
        :return: list of all models
        """
        return [node.model for node in sorted(self.nodes, key=lambda x: x.node_id)]

    @property
    def problems(self) -> List[OptimalControlProblem]:
        """
            List of problems of all nodes

        :rtype: list of OptimalControlProblem
        :return: list of all problems
        """
        return [node.problem for node in sorted(self.nodes, key=lambda x: x.node_id)]

    def create_node(
        self,
        name: Optional[str] = None,
        model: Optional[SystemModel] = None,
        problem: Optional[OptimalControlProblem] = None,
        **kwargs: Any,
    ):
        """
            Creat a node in the network

        :param str name:
        :param SystemModel model:
        :param OptimalControlProblem problem:

        :rtype: Node
        :return: node
        """
        node = Node(name=name, model=model, problem=problem, **kwargs)
        self.include_nodes(node)

        return node

    def include_nodes(self, nodes):
        """
            Include node or list of nodes in the network

        :param list|Node nodes: node/nodes
        """
        if not isinstance(nodes, list):
            nodes = [nodes]

        # if node does not have an id
        for node in nodes:
            if node.node_id is None:
                self._set_node_an_id(node)

        self.graph.add_nodes_from(nodes)

    def remove_connection(self, node1, node2):

        """
            Remove a connection between node1 and node2

        :param Node node1:
        :param Node node2:
        """
        self.graph.remove_edge(node1, node2)

    def connect(self, y: SX, u: SX, node1: Node, node2: Node):
        """
            Connect the variables of two subsystems (nodes) by creating an edge

        :param y: output variable of node1
        :param u: input variable of node2
        :param node1: node1 (which has 'y')
        :param node2: node2 (which has 'u')
        """
        if not node1.model.has_variable(y):
            raise ValueError(
                '"node1" ({}) does not have the passed "y" ({})'.format(node1.name, y)
            )
        if not node2.model.has_variable(u):
            raise ValueError(
                '"node2" ({}) does not have the passed "u" ({})'.format(node2.name, y)
            )

        if not self.intermediary_nodes:
            self._connect(y, u, node1, node2)
        else:
            self.insert_intermediary_node(y, u, node1, node2)

    def _connect(self, y: SX, u: SX, node1: Node, node2: Node):
        if (node1, node2) not in self.graph.edges:
            self.graph.add_edge(node1, node2, y=DM(), u=DM())

        self.graph.edges[node1, node2]["y"] = vertcat(
            self.graph.edges[node1, node2]["y"], y
        )
        self.graph.edges[node1, node2]["u"] = vertcat(
            self.graph.edges[node1, node2]["u"], u
        )

    def get_model(self) -> SystemModel:
        """
            Create a single model which is the composition of all models and the connections

        :return: model
        :rtype: SystemModel
        """
        name = self.name + "_model" if self.name is not None else "Network"
        model = SystemModel(name=name)
        model.include_models(self.models)
        for edge in self.graph.edges:
            model.connect(u=self.graph.edges[edge]["u"], y=self.graph.edges[edge]["y"])

        return model

    def get_problem(self):
        """
            Create a single OCP which is the composition of all problems and connections.

        :return: problem
        :rtype: OptimalControlProblem
        """
        model = SystemModel(name=self.name + "_model")
        problem = OptimalControlProblem(model=model, name=self.name + "_problem")
        problem.t_0 = self.problems[0].t_0
        problem.t_f = self.problems[0].t_f
        problem.merge(self.problems)
        for edge in self.graph.edges:
            problem.connect(
                u=self.graph.edges[edge]["u"],
                y=self.graph.edges[edge]["y"],
                replace=False,
            )

        return problem

    def get_node_by_id(self, id: int):
        for node in self.graph:
            if node.node_id == id:
                return node

    def get_node_by_name(self, name: str):
        for node in self.graph:
            if node.name == name:
                return node

    def _set_node_an_id(self, node):
        """
            Set a unique id to a node

        :param Node node:
        :return: node id
        :rtype: int
        """
        node.node_id = self._nodes_id_counter
        self._nodes_id_counter += 1

        return node.node_id

    def plot(self):
        """
        Plot the network
        """
        labels = dict([(node, node.name) for node in self.nodes])

        colors = [node.color for node in self.nodes]

        networkx.draw_circular(
            self.graph,
            node_size=2000,
            node_color=colors,
            with_labels=True,
            labels=labels,
            font_weight="bold",
            font_color="white",
        )
        plt.show()

    def get_map_coloring_groups(self):
        coloring_dict = networkx.greedy_color(self.graph.to_undirected())
        grouping_dict = {}

        for node in coloring_dict:
            if coloring_dict[node] not in grouping_dict:
                grouping_dict[coloring_dict[node]] = []
            grouping_dict[coloring_dict[node]].append(node)

        return list(grouping_dict.values())

    def insert_intermediary_nodes(self):
        old_connections = {
            (node1, node2): (
                self.graph.edges[node1, node2]["y"],
                self.graph.edges[node1, node2]["u"],
            )
            for (node1, node2) in self.graph.edges
        }
        self.graph = networkx.DiGraph()

        for (node1, node2), (y, u) in old_connections.items():
            self.insert_intermediary_node(y, u, node1, node2)

    def insert_intermediary_node(self, y, u, node1, node2):
        new_model = SystemModel(
            name="Dummy_Model_{}_to_{}".format(node1.name, node2.name)
        )
        dummy_u = vertcat(
            *[new_model.create_control(f"Dummy_{var.name()}") for var in y.nz]
        )
        dummy_y = vertcat(
            *[
                new_model.create_algebraic_variable(f"Dummy_{var.name()}")
                for var in u.nz
            ]
        )

        new_model.include_equations(alg=[dummy_u - dummy_y])

        new_problem = OptimalControlProblem(
            name="OCP_Dummy_{}_to_{}".format(node1.name, node2.name),
            model=new_model,
            t_f=node1.problem.t_f,
        )

        new_node = self.create_node(
            name="Dummy_node_{}_to_{}".format(node1.name, node2.name),
            model=new_model,
            problem=new_problem,
            color=0.75,
        )

        self._connect(y, dummy_u, node1, new_node)
        self._connect(dummy_y, u, new_node, node2)

    def get_initial_condition(self, t_f):
        initial_condition = {}
        problem = self.get_problem()
        final_condition = problem.simulate(t_f=t_f).final_condition()

        def find_type_and_index(var):
            if index := find_variables_indices_in_vector(var, problem.model.x):
                return "x", index
            if index := find_variables_indices_in_vector(var, problem.model.y):
                return "y", index
            if index := find_variables_indices_in_vector(var, problem.model.u):
                return "u", index
            raise ValueError(f"Could not find variable, {var.name()}")

        def get_variable_final_condition(final_condition, var):
            var_type, index = find_type_and_index(var)
            return final_condition[var_type][index]

        for node in self.nodes:
            initial_condition[node.name] = {
                "x": DM(
                    [
                        get_variable_final_condition(final_condition, x)
                        for x in node.model.x.nz
                    ]
                ),
                "y": DM(
                    [
                        get_variable_final_condition(final_condition, y)
                        for y in node.model.y.nz
                    ]
                ),
                "u": DM(
                    [
                        get_variable_final_condition(final_condition, u)
                        for u in node.model.u.nz
                    ]
                ),
            }

        return initial_condition
