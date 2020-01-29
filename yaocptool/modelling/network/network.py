# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:05:50 2017

@author: marco
"""

import matplotlib.pyplot as plt
import networkx
from casadi import vertcat, SX

from yaocptool import find_variables_indices_in_vector, DM
from yaocptool.modelling import SystemModel, OptimalControlProblem
from yaocptool.modelling.network.node import Node


class Network:
    def __init__(self, nodes=None, name='network'):
        """

        :param list of Node nodes:
        """

        self.name = name

        self.graph = networkx.DiGraph()  # type: networkx.DiGraph
        self._nodes_id_counter = 0

        if nodes is not None:
            self.include_nodes(nodes)

    @property
    def nodes(self):
        """
            All nodes (from the self.graph)

        :rtype: networkx.reportviews.NodeView
        """
        return self.graph.nodes

    @property
    def connections(self):
        """
            All connections (edges) from the self.graph

        :rtype: networkx.reportviews.OutEdgeView
        """
        return self.graph.edges

    @property
    def models(self):
        """
            List of models of all nodes

        :rtype: list of SystemModel
        :return: list of all models
        """
        models = []
        for node in sorted(self.nodes, key=lambda x: x.node_id):
            models.append(node.model)
        return models

    @property
    def problems(self):
        """
            List of problems of all nodes

        :rtype: list of OptimalControlProblem
        :return: list of all problems
        """
        problems = []
        for node in sorted(self.nodes, key=lambda x: x.node_id):
            problems.append(node.problem)
        return problems

    def create_node(self, name=None, model=None, problem=None, **kwargs):
        """
            Creat a node in the network

        :param str name:
        :param SystemModel model:
        :param OptimalControlProblem problem:

        :rtype: Node
        :return: node
        """
        node = Node(name=name, model=model, problem=problem, **kwargs)
        self._set_node_an_id(node)

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

    def connect(self, y, u, node1, node2):
        """
            Connect the variables of two subsystems (nodes) by creating an edge

        :param y: output variable of node1
        :param u: input variable of node2
        :param node1: node1 (which has 'y')
        :param node2: node2 (which has 'u')
        """
        if not node1.model.has_variable(y):
            raise ValueError('"node1" ({}) does not have the passed "y" ({})'.format(node1.name, y))
        if not node2.model.has_variable(u):
            raise ValueError('"node2" ({}) does not have the passed "u" ({})'.format(node2.name, y))
        if not (node1, node2) in self.graph.edges:
            self.graph.add_edge(node1, node2, y=DM([]), u=DM([]))

        self.graph.edges[node1, node2]['y'] = vertcat(self.graph.edges[node1, node2]['y'], y)
        self.graph.edges[node1, node2]['u'] = vertcat(self.graph.edges[node1, node2]['u'], u)

    def remove_connection(self, node1, node2):
        """
            Remove a connection between node1 and node2

        :param Node node1:
        :param Node node2:
        """
        self.graph.remove_edge(node1, node2)

    def get_model(self):
        """
            Create a single model which is the composition of all models and the connections

        :return: model
        :rtype: SystemModel
        """
        model = SystemModel(name=self.name + '_model')
        model.include_models(self.models)
        for edge in self.graph.edges:
            model.connect(u=self.graph.edges[edge]['u'],
                          y=self.graph.edges[edge]['y'])

        return model

    def get_problem(self):
        """
            Create a single OCP which is the composition of all problems and connections.

        :return: problem
        :rtype: OptimalControlProblem
        """
        model = SystemModel(name=self.name + '_model')
        problem = OptimalControlProblem(model=model, name=self.name + '_problem')
        problem.t_0 = self.problems[0].t_0
        problem.t_f = self.problems[0].t_f
        problem.merge(self.problems)
        for edge in self.graph.edges:
            problem.connect(u=self.graph.edges[edge]['u'],
                            y=self.graph.edges[edge]['y'])

        return problem

    def get_node_by_id(self, id):
        for node in self.nodes:
            if node.node_id == id:
                return node

    def get_node_by_name(self, name):
        for node in self.nodes:
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
        self._nodes_id_counter = self._nodes_id_counter + 1

        return node.node_id

    def plot(self):
        """
            Plot the network
        """
        labels = dict([(node, node.name) for node in self.nodes])

        colors = [node.color for node in self.nodes]

        networkx.draw_spring(self.graph,
                             node_size=2000, node_color=colors,
                             with_labels=True, labels=labels,
                             font_weight='bold', font_color='white')
        plt.show()

    def get_map_coloring_groups(self):
        coloring_dict = networkx.greedy_color(self.graph.to_undirected())
        grouping_dict = {}

        for node in coloring_dict:
            if not coloring_dict[node] in grouping_dict:
                grouping_dict[coloring_dict[node]] = []
            grouping_dict[coloring_dict[node]].append(node)

        return list(grouping_dict.values())

    def insert_intermediary_nodes(self):
        old_connections = list(self.connections)

        for (node1, node2) in old_connections:
            y = self.graph.edges[node1, node2]['y']
            u = self.graph.edges[node1, node2]['u']

            y_guess = vertcat(*node1.problem.y_guess)[find_variables_indices_in_vector(y, node1.problem.model.y)]
            u_guess = vertcat(*node2.problem.u_guess)[find_variables_indices_in_vector(u, node2.problem.model.u)]

            copy_y = vertcat(*[SX.sym('Dummy_' + y[ind].name()) for ind in range(y.numel())])
            copy_u = vertcat(*[SX.sym('Dummy_' + u[ind].name()) for ind in range(u.numel())])

            new_model = SystemModel(name='Dummy_Model_{}_to_{}'.format(node1.name, node2.name))
            new_model.include_variables(u=copy_y, y=copy_u)
            new_model.include_equations(alg=copy_u - copy_y)
            new_problem = OptimalControlProblem(name='OCP_Dummy_{}_to_{}'.format(node1.name, node2.name),
                                                model=new_model, t_f=node1.problem.t_f,
                                                y_guess=u_guess, u_guess=y_guess)

            new_node = Node(name='Dummy_node_{}_to_{}'.format(node1.name, node2.name),
                            model=new_model, problem=new_problem, color=0.75)

            self.include_nodes(new_node)

            self.remove_connection(node1, node2)
            self.connect(y, copy_y, node1, new_node)
            self.connect(copy_u, u, new_node, node2)

        for node in self.nodes:
            print(node.node_id, node.name)
