from tests.models import create_2x2_mimo
from yaocptool.methods import DirectMethod

nlpsol_opts = {"ipopt.print_level": 0, "print_time": False}

# test_direct_collocation_polynomial_control
model, problem = create_2x2_mimo()
solution_method = DirectMethod(
    problem,
    degree=3,
    degree_control=3,
    finite_elements=20,
    discretization_scheme="collocation",
    nlpsol_opts=nlpsol_opts,
)
result = solution_method.solve()
print(result.objective_opt_problem)
