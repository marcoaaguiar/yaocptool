from __future__ import print_function

import multiprocessing


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
