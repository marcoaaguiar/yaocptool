# -*- coding: utf-8 -*-
"""
Created on 

@author: Marco Aurelio Schmitz de Aguiar
"""


class SolutionMethodInterface:
    def solve(self):
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
