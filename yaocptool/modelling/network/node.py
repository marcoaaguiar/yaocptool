# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:33:52 2017

@author: marco
"""
from typing import Optional
from yaocptool.modelling import SystemModel, OptimalControlProblem


class Node:
    def __init__(
        self,
        node_id: int = None,
        name: Optional[str] = None,
        model: Optional[SystemModel] = None,
        problem: Optional[OptimalControlProblem] = None,
        color: float = 0.25,
    ):
        self.name = name
        self.node_id = node_id
        self.problem = problem
        self.color = color

        self.model = problem.model if model is None and problem is not None else model

        if name is None and self.model is not None:
            self.name = self.model.name

    def __repr__(self) -> str:
        return "<Node name: {}, node_id: {}>".format(self.name, self.node_id)
