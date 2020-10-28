from yaocptool.modelling.ocp import OptimalControlProblem
from casadi.casadi import Function, MX, horzcat


class IntermediaryNodeSolutionMethod:
    degree = 3
    finite_elements = 10

    def __init__(self, problem: OptimalControlProblem):
        self.problem = problem

    def _create_solve_function(self):
        mu_k = MX.sym("mu_k")

        y_p1 = [
            [MX.sym("y_p1_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]
        u_p2 = [
            [MX.sym("u_p2_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]

        nu_p1_c = [
            [MX.sym("nu_p1_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]
        nu_c_p2 = [
            [MX.sym("nu_p2_{el}_{deg}") for deg in range(self.degree)]
            for el in range(self.finite_elements)
        ]

        vectorize = lambda var: horzcat(
            *[horzcat(*var[el]) for el in range(self.finite_elements)]
        )
        y_p1, u_p2, nu_p1_c, nu_c_p2 = (
            vectorize(y_p1),
            vectorize(u_p2),
            vectorize(nu_p1_c),
            vectorize(nu_c_p2),
        )
        u_c = (nu_c_p2 - nu_p1_c) / (2 * mu_k) + (y_p1 + u_p2) / 2

        u_c_function = Function("f_u_c", [mu_k, y_p1, u_p2, nu_p1_c, nu_c_p2], [u_c])

    def solve(self, *args, **kwargs):
        pass


if __name__ == "__main__":
    problem = None
    sol = IntermediaryNodeSolutionMethod(problem)
    sol._create_solve_function()
