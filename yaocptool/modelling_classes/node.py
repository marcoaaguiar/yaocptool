# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:33:52 2017

@author: marco
"""


class Node:
    def __init__(self, node_id=-1, name='', model=None, problem=None, **kwargs):
        assert node_id != -1
        self.name = name
        self.node_id = node_id

        self.problem = problem
        if model is None and problem is not None:
            self.model = self.problem.model
        else:
            self.model = model

        if name == '' and hasattr(self.model, 'name'):
            self.name = self.model.name

        self.connected_nodes = {}
