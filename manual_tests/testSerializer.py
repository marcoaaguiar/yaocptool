import concurrent

from casadi import DM

from yaocptool.optimization import NonlinearOptimizationProblem

if __name__ == "__main__":
    size = 4
    nlp = NonlinearOptimizationProblem()
    x = nlp.create_variable("x", size)
    nlp.set_objective(x.T @ x)
    nlp.include_constraint(x <= DM(range(size)))

    with concurrent.futures.ProcessPoolExecutor() as executor:
        future = executor.submit(nlp.solve, 4)
        print(future.result())
