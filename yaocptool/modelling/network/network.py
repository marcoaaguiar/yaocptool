# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:05:50 2017

@author: marco
"""
import matplotlib.pyplot as plt
import networkx
from casadi import vertcat

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
        self.graph.add_nodes_from(nodes)

    def connect(self, y, u, node1, node2):
        """
            Connect the variables of two subsystems (nodes) by creating an edge

        :param y: output variable of node1
        :param u: input variable of node2
        :param node1: node1 (which has 'y')
        :param node2: node2 (which has 'u')
        """
        if not (node1, node2) in self.graph.edges:
            self.graph.add_edge(node1, node2, y=vertcat([]), u=vertcat([]))

        self.graph.edges[node1, node2]['y'] = vertcat(self.graph.edges[node1, node2]['y'], y)
        self.graph.edges[node1, node2]['u'] = vertcat(self.graph.edges[node1, node2]['u'], u)

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

        networkx.draw_circular(self.graph,
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
