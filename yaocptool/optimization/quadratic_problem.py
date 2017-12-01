from casadi import qpsol

from yaocptool.optimization.abstract_optimization_problem import AbstractOptimizationProblem


class QuadraticOptimizationProblem(AbstractOptimizationProblem):
    def __init__(self, n_x=0, **kwargs):
        """ Abstract Optimization Problem class
            Optimization problem

            minimize f(x)
               x
            subject to: g_eq = 0
                        g_ineq <= 0
            Object attributes:
            x -> optimization variables
            g -> contraint

        :param n_x: int
        """
        super(QuadraticOptimizationProblem, self).__init__(n_x, **kwargs)

    def _create_solver(self):
        problem_dict = self.get_problem_dict()
        return qpsol(self.name + '_solver', 'qpoases', problem_dict, self.solver_options)
