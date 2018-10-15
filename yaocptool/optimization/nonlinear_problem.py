from casadi import nlpsol

from yaocptool.config import SOLVER_OPTIONS
from yaocptool.optimization.abstract_optimization_problem import AbstractOptimizationProblem


class NonlinearOptimizationProblem(AbstractOptimizationProblem):
    def __init__(self, **kwargs):
        """
            Nonlinear Optimization Problem class
            Optimization problem

        .. math::
            \\min_x f(x, p)

            \\textrm{s.t.:} g_{lb} \leq g(x,p) \leq g_{ub}

        Object attributes:
        x -> optimization variables
        g -> constraint
        """
        super(NonlinearOptimizationProblem, self).__init__(**kwargs)
        if self.solver_options == {}:
            self.solver_options = SOLVER_OPTIONS['nlpsol_options']

    def _create_solver(self):
        problem_dict = self.get_problem_dict()
        return nlpsol(self.name + '_' + 'solver', 'ipopt', problem_dict, self.solver_options)
