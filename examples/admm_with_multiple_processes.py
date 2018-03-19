"""This example shows how to use the parallel.Worker class.
The following NLP will be used as an example

 \min x^2 + exp(-z)
s.t.: x=z

Applying Augmented Lagrangian, we get

\min x^2 + exp(-z) + \lambda(x-z) + \mu/2*(x-z)^2

The ADMM steps are:

1. Solve \min x^2 + \lambda(x-z) + \mu/2*(x-z)^2, with z fixed.

2. Solve \min exp(-z) + \lambda(x-z) + \mu/2*(x-z)^2, with x fixed.

3. Update \lambda_{k+1} = \lambda_k + \mu (x-z) and Update \mu_{k+1} = \beta*\mu_k, with \beta>1

"""
from multiprocessing import Queue

import time
from casadi import exp, vertcat
from matplotlib import pyplot

from yaocptool.optimization import NonlinearOptimizationProblem
from yaocptool.parallel.worker import Worker

LAMBDA_0 = 1.
MU_0 = 1.
MAX_ITERATIONS = 10


# Create first problem
class Problem1:
    def __init__(self):
        nlp = NonlinearOptimizationProblem()
        x = nlp.create_variable('x')
        z = nlp.create_parameter('z')
        mu = nlp.create_parameter('mu')
        lamb = nlp.create_parameter('lambda')
        obj = x ** 2 + lamb * (x - z) + mu / 2 * (x - z) ** 2
        nlp.set_objective(obj)
        self.nlp = nlp

        self.iteration = 0
        self.max_iterations = MAX_ITERATIONS
        self.stop = False

    def solve(self, z, mu_lamb):
        mu, lamb = mu_lamb
        sol = self.nlp.solve(p=vertcat(z, mu, lamb))

        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.stop = True

        return sol['x']


# Create second problem
class Problem2:
    def __init__(self):
        nlp = NonlinearOptimizationProblem()
        z = nlp.create_variable('z')
        x = nlp.create_parameter('x')
        mu = nlp.create_parameter('mu')
        lamb = nlp.create_parameter('lambda')
        obj = exp(-z) + lamb * (x - z) + mu / 2 * (x - z) ** 2
        nlp.set_objective(obj)
        self.nlp = nlp

        self.iteration = 0
        self.max_iterations = MAX_ITERATIONS
        self.stop = False

    def solve(self, x, mu_lamb):
        mu, lamb = mu_lamb
        sol = self.nlp.solve(p=vertcat(x, mu, lamb))

        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.stop = True

        return sol['x']


# Create an Updater Class, which performs the third step of the ADMM
class Updater:
    def __init__(self):
        self.lamb = LAMBDA_0
        self.mu = MU_0

        self.iteration = 0
        self.max_iterations = MAX_ITERATIONS
        self.stop = False

    def update(self, nlp1_sol, nlp2_sol):
        x = nlp1_sol
        z = nlp2_sol

        new_lamb = self.lamb + self.mu * (x - z)
        self.lamb = new_lamb
        self.mu = 2 * self.mu

        self.iteration += 1
        if self.iteration >= self.max_iterations:
            self.stop = True

        return self.mu, self.lamb


if __name__ == '__main__':
    # Create the Queues for communication between processes
    queue12 = Queue()
    queue13 = Queue()
    queue23 = Queue()

    queue21 = Queue()
    queue31 = Queue()
    queue32 = Queue()

    # Queues to get the result of each process
    queue1_listener = Queue()
    queue2_listener = Queue()
    queue3_listener = Queue()

    # Create the Worker which will spawn a new process creating an object Problem1/Problem2/Updater, and calling a
    # designated function after getting one element from each queue
    w_nlp1 = Worker(Problem1, None, 'solve', [queue21, queue31], [queue12, queue13, queue1_listener])
    w_nlp2 = Worker(Problem2, None, 'solve', [queue12, queue32], [queue21, queue23, queue2_listener])
    w_update = Worker(Updater, None, 'update', [queue13, queue23], [queue31, queue32, queue3_listener])

    # We need to initialize some queues in order to some of the process start
    queue21.put(0.)
    queue31.put([MU_0, LAMBDA_0])
    queue32.put([MU_0, LAMBDA_0])

    # Start the processes
    w_nlp1.start()
    w_nlp2.start()
    w_update.start()

    # Get the results
    nlp1_sol = []
    nlp2_sol = []
    update_res = []
    for it in range(MAX_ITERATIONS):
        nlp1_sol.append(queue1_listener.get(block=True))
        nlp2_sol.append(queue2_listener.get(block=True))
        update_res.append(queue3_listener.get(block=True))

    # Print the result of each iteration
    print(nlp1_sol)
    print(nlp2_sol)
    print(update_res)

    pyplot.plot(nlp1_sol, nlp2_sol, 'x-')
