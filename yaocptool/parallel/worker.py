from __future__ import print_function
import multiprocessing
import sys

from casadi import SX, DM, Function, nlpsol, dot, vertcat, tan, sum1
import time


class Worker(multiprocessing.Process):
    """Creates new process that creates and object of class 'obj_class' with 'obj_arg' argument.
    It will consume one element from each queue_in and call function 'function_name' the consumed elements as argument.
    It will put the return of the 'function_name' call in all Queues in queue_out

    """

    def __init__(self, obj_class, obj_arg, function_name, queue_in, queue_out):
        multiprocessing.Process.__init__(self)

        if not isinstance(queue_in, list):
            queue_in = [queue_in]
        if not isinstance(queue_out, list):
            queue_out = [queue_out]

        self.obj_class = obj_class
        self.obj_arg = obj_arg
        self.function_name = function_name
        self.queue_in = queue_in
        self.queue_out = queue_out
        self.obj = None

    def run(self):
        print('Starting to run: %s' % self.name)
        if self.obj_arg is not None:
            self.obj = self.obj_class(self.obj_arg)
        else:
            self.obj = self.obj_class()

        while True:
            # Check if object is trying to stop
            if hasattr(self.obj, 'stop'):
                if self.obj.stop:
                    break

            data = [queue.get(block=True) for queue in self.queue_in]
            result = getattr(self.obj, self.function_name)(*data)

            for queue in self.queue_out:
                queue.put(result)
        # self.terminate()


if __name__ == '__main__':
    # Target object, creating here just as an example
    class Task:
        def __init__(self, i):
            self.size = 10000
            p = SX.sym('p')
            x = SX.sym('x', self.size)
            nlp = {'f': dot(x - p, x - p), 'x': x, 'g': vertcat(dot(x, x) - 1, sum1(tan(x))), 'p': p}
            self.solver = nlpsol('nlp', 'ipopt', nlp, {'print_time': False, 'ipopt': {'print_level': 0}})

        def call(self, p_value):
            res = self.solver(x0=DM.zeros(self.size), lbx=-DM.inf(self.size), ubx=DM.inf(self.size),
                              ubg=vertcat(0, DM.inf(1)), lbg=vertcat(-DM.inf(1), 1), p=p_value)
            return res


    # Parallel
    t1 = time.time()
    workers = []
    queue_out = multiprocessing.Queue()
    queue_in = multiprocessing.Queue()

    # Create the workers
    num_workers = 4
    for i in range(num_workers):
        w = Worker(Task, i, 'call', queue_in, queue_out)
        workers.append(w)
        w.start()

    # put data to be processed in the input queue
    par_range = 50
    for j in range(par_range):
        queue_in.put(j)

    # Wait all the 50 data to be processed
    a = [queue_out.get() for i in range(par_range)]
    dt1 = time.time() - t1
    print('parallel: ', dt1)

    # Same thing but in parallel
    # if False:
    if True:
        t2 = time.time()
        task = Task(i)
        for k in range(par_range):
            task.call(k)
        dt2 = time.time() - t2
        print('serial: ', dt2)
