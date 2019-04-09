# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:33:52 2017

@author: marco
"""


class Node:
    def __init__(self, node_id=None, name='', model=None, problem=None, color=0.25, **kwargs):
        """

        :param int node_id:
        :param str name:
        :param yaocptool.modelling.SystemModel model:
        :param yaocptool.modelling.OptimalControlProblem problem:
        :param float color:
        :param dict kwargs:
        """
        self.name = name
        self.node_id = node_id
        self.problem = problem
        self.color = color

        if model is None and problem is not None:
            self.model = self.problem.model
        else:
            self.model = model

        if name == '' and hasattr(self.model, 'name'):
            self.name = self.model.name

        self.connected_nodes = {}

        for (k, v) in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return '<Node name: {}, node_id: {}>'.format(self.name, self.node_id)
