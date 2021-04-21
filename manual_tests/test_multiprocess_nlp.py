import asyncio
import concurrent.futures
from multiprocessing import Pipe, Pool, Process
from multiprocessing.connection import Connection
from typing import List

import ray
from casadi import DM, Function

from yaocptool import Timer
from yaocptool.optimization import NonlinearOptimizationProblem


def make_nlps(n: int = 100, size: int = 10000) -> List[NonlinearOptimizationProblem]:
    nlps = []
    for _ in range(n):
        nlp = NonlinearOptimizationProblem()
        x = nlp.create_variable("x", size)
        #  p = nlp.create_parameter("p")
        nlp.set_objective(x.T @ x)
        nlp.include_constraint(x <= DM(range(size)))
        nlps.append(nlp)
    return nlps


if __name__ == "__main__":
    # serial
    loops = 20
    nlps = make_nlps()
    for nlp in nlps:
        nlp.get_solver()
    with Timer(verbose=True) as serial_timer:
        for _ in range(loops):
            result = [nlp.solve(None) for nlp in nlps]

    #  with multiprocessing
    print("multiprocessing")
    nlps = make_nlps()
    for nlp in nlps:
        nlp.get_solver()
    with Timer(verbose=True) as mp_timer:
        for _ in range(loops):
            for nlp in nlps:
                nlp.mp_solve(None)

            result = [nlp.mp_get_solution() for nlp in nlps]

        for nlp in nlps:
            nlp.mp_terminate()

    # concurrent
    print("concurrent")
    nlps = make_nlps()
    with Timer(verbose=True) as conc_timer:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in range(loops):
                sols = [executor.submit(nlp.solve, None) for nlp in nlps]
                result = [sol.result() for sol in sols]

    # concurrent pre init
    nlps = make_nlps()
    for nlp in nlps:
        nlp.get_solver()
    with Timer(verbose=True) as conc_timer_pre_init:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for _ in range(loops):
                sols = [executor.submit(nlp.solve, None) for nlp in nlps]
                result = [sol.result() for sol in sols]

    # ray
    #  ray.init()
    #  nlps = make_nlps()
    #  for nlp in nlps:
    #      nlp.get_solver()
    #  remote_nlp = [ray.put(nlp) for nlp in nlps]
    #
    #  with Timer(verbose=True) as ray_timer:
    #      for _ in range(loops):
    #          futures = [ray_solve_nlp.remote(nlp) for nlp in remote_nlp]
    #          result = ray.get(futures)

    #  nlps = make_nlps()
    #  for nlp in nlps:
    #      nlp.get_solver()
    #
    #
    #  async def main(nlps):
    #      loop = asyncio.get_running_loop()
    #      result = None
    #      with concurrent.futures.ProcessPoolExecutor() as pool:
    #          for _ in range(loops):
    #              result = await asyncio.gather(
    #                  *[loop.run_in_executor(pool, nlp.solve, None) for nlp in nlps]
    #              )
    #      return result

    #  with Timer(verbose=True) as asyn_timer:
    #      asyncio.run(main(nlps))

    print("seria", serial_timer.elapsed)
    print("mp", mp_timer.elapsed)
    print("concurrent", conc_timer.elapsed)
    print("concurrent pre init", conc_timer_pre_init.elapsed)
    #  print("ray", ray_timer.elapsed)
    #  print("async", asyn_timer.elapsed)
