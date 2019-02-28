# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:05:50 2017

@author: marco
"""

from casadi import vertcat, depends_on


class Network:
    def __init__(self, nodes, connections_settings_dict):
        """
        Example of connections_dict:
            Connections 0:
            z_sym[0] in Node 0 is connected to z_sym[2] in Node 1
            connections_dict = {0: [{0:[0]}, {1:[2]}]}
        it can also be a multidimensional connection:
            Connection 1:
            z_sym[0:1] in Node 1 is connected to z_sym[1:2] in Node 4
            connections_dict = {1: [{1:[0,1]}, {4:[1,2]}]}

        :param nodes:
        :param connections_settings_dict:
        """
        self.nodes = nodes
        self.connections_settings_dict = connections_settings_dict

        self.nodes_dict = {}
        for node in self.nodes:
            self.nodes_dict[node.node_id] = node

        self.connection_dict = {}
        self.create_connections()

    def create_connections(self):
        for connection_id in self.connections_settings_dict:
            nodes = [self.nodes_dict[node_id] for node_id in self.connections_settings_dict[connection_id]]
            z_sym_indices_list = self.connections_settings_dict[connection_id].values()
            self.connection_dict[connection_id] = Connection(connection_id=connection_id, nodes=nodes,
                                                             z_sym_indices_list=z_sym_indices_list)

    @property
    def models(self):
        models = []
        for node in self.nodes:
            models.append(node.model)
        return models

    @property
    def problems(self):
        problems = []
        for node in self.nodes:
            problems.append(node.problem)
        return problems

    def get_connection_equations(self):
        eqs = []
        for connection in self.connection_dict.values():
            eqs.append(connection.equation)
        return vertcat(*eqs)

    def get_connection_defined_z(self):
        zs = []
        for connection in self.connection_dict.values():
            zs.append(connection.defined_z_sym)
        return vertcat(*zs)


class Connection:
    def __init__(self, connection_id, nodes=None, z_sym_indices_list=None):
        """
            Connection class for Network systems

        :param list nodes:
        :param dict z_sym_indices_list:
        :param int connection_id:
        """
        if z_sym_indices_list is None:
            z_sym_indices_list = {}
        if nodes is None:
            nodes = []
        self.connection_id = connection_id
        self.nodes = nodes
        self.associated_z_sym = [nodes[0].model.z_sym[z_sym_indices_list[0]],
                                 nodes[1].model.z_sym[z_sym_indices_list[1]]]

        self.equation = vertcat(self.associated_z_sym[0] - self.associated_z_sym[1])
        for node in self.nodes:
            for z_sym in self.associated_z_sym:
                if not depends_on(node.model.alg_z, z_sym):
                    self.defined_z_sym = z_sym
