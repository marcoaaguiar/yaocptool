# -*- coding: utf-8 -*-
"""
Created on

@author: Marco Aurelio Schmitz de Aguiar
"""


class SolutionMethodInterface(object):
    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def solve(self, *args, **kwargs):
        """
        Method that will solve the yaoctool.modelling.OptimalControlProblem and return a
        yaocptool.modelling.OptimizationResult,
        """
        raise NotImplementedError

    def prepare(self):
        """
        Perform pre solve check, check for sizes and types. If it fails raise an Exception
        """
        raise NotImplementedError
