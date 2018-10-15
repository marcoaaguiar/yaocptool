from casadi import qpsol

from yaocptool.optimization.abstract_optimization_problem import AbstractOptimizationProblem


class QuadraticOptimizationProblem(AbstractOptimizationProblem):
    def __init__(self, **kwargs):
        """
            Quadratic Optimization Problem class
            Optimization problem

        .. math::
            \\min_x &f(x, p)

            \\textrm{s.t.:} &g_{lb} \leq g(x,p) \leq g_{ub}

        Object attributes:
        x -> optimization variables
        g -> constraint
        """
        super(QuadraticOptimizationProblem, self).__init__(**kwargs)

    def _create_solver(self):
        problem_dict = self.get_problem_dict()
        return qpsol(self.name + '_solver', 'qpoases', problem_dict, self.solver_options)
