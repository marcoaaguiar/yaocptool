from multiprocessing import Queue

from yaocptool.parallel.worker import Worker


class DistributedSolution:
    def __init__(self, subsystem_classes_list, parameters_list, connection_list, main_to_subsystems_list=None, **kwargs):
        if main_to_subsystems_list is None:
            main_to_subsystems_list = []

        self.subsystem_classes_list = subsystem_classes_list
        self.parameters_list = parameters_list
        self.connection_list = connection_list

        # Queues between subsystems
        self.queues_in = dict([(ind, {}) for ind in range(self.n_subsystems)])
        self.queues_out = dict([(ind, {}) for ind in range(self.n_subsystems)])

        # Queue between main process and subsystems and vice-versa
        self.external_queue = {}
        self.queue_listener = {}

        # Create the queue between main process and subsystems
        for s in main_to_subsystems_list:
            queue = Queue()
            self.external_queue[s] = queue

        # Create queues between systems
        for connection in connection_list:
            from_index, to_index = connection
            queue = Queue()
            self.queues_in[to_index][from_index] = queue
            self.queues_out[from_index][to_index] = queue

        # Create queues between subsystems and main process
        for s in range(self.n_subsystems):
            queue = Queue()
            self.queue_listener[s] = queue

        self.workers = []
        for s, subsystem_class in enumerate(subsystem_classes_list):
            queue_in = []
            if s in self.external_queue:
                queue_in += [self.external_queue[s]]
            queue_in += self.queues_in[s].values()

            worker = Worker(subsystem_class, parameters_list[s], 'solve',
                            queue_in=queue_in,
                            queue_out=self.queues_out[s].values() + [self.queue_listener[s]])
            self.workers.append(worker)

        self.initialize()
        self.start()

    @property
    def n_subsystems(self):
        return len(self.subsystem_classes_list)

    def start(self):
        for worker in self.workers:
            worker.start()

    def initialize(self):
        for s in range(self.n_subsystems):
            for r in range(s, self.n_subsystems):
                if (r, s) in self.connection_list:
                    self.queues_in[s][r].put(None)

    def solve(self, x_0, initial_guess_dict=None):
        if initial_guess_dict is None:
            initial_guess_dict = [None] * self.n_subsystems

        for s in range(self.n_subsystems):
            self.external_queue[s].put([x_0, initial_guess_dict[s]])

        solution = []
        for s in range(self.n_subsystems):
            solution.append(self.queue_listener[s].get(block=True))
        return solution
