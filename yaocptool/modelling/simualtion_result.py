from yaocptool.modelling import DataSet

try:
    import matplotlib.pyplot as plt
except ImportError:
    print('Failed to import matplotlib. Make sure that it is properly installed')
    plt = None
from casadi import horzcat, vertcat


class SimulationResult(DataSet):
    def __init__(self, **kwargs):
        self.model_name = ''
        self.delta_t = 1
        self.t_0 = 0
        self.t_f = 1
        self.finite_elements = 0

        DataSet.__init__(self, **kwargs)

    @property
    def x(self):
        return [self.data['x']['values'][:, i] for i in range(self.data['x']['values'].shape[1])]

    @property
    def y(self):
        return [self.data['y']['values'][:, i] for i in range(self.data['y']['values'].shape[1])]

    @property
    def u(self):
        return [self.data['u']['values'][:, i] for i in range(self.data['u']['values'].shape[1])]

    @property
    def t(self):
        return self.data['x']['times']

    @property
    def n_x(self):
        return self.data['x']['size']

    @property
    def n_y(self):
        return self.data['y']['size']

    @property
    def n_u(self):
        return self.data['u']['size']

    @property
    def x_names(self):
        return self.data['x']['names']

    @property
    def y_names(self):
        return self.data['y']['names']

    @property
    def u_names(self):
        return self.data['u']['names']

    def final_condition(self):
        """Return the simulation final condition in the form of a tuple (x_f, y_f, u_f)

        :rtype: DM, DM, DM
        """
        return self.data['x']['values'][:, -1], self.data['y']['values'][:, -1], self.data['u']['values'][:, -1]

    def extend(self, other_sim_result):
        """Extend this SimulationResult with other SimulationResult.
        It is only implemented for the case where the other simulation result starts at the end of this
        simulation. That is this.t_f == other_sim_result.t_0 .

        :param SimulationResult other_sim_result:
        :return:
        """
        list_of_attributes_to_check = ['n_x', 'n_y', 'n_u']
        for attr in list_of_attributes_to_check:
            if not getattr(self, attr) == getattr(other_sim_result, attr):
                raise Exception(
                    "Attribute {} is no equal for both SimulationResults: {}!={}".format(attr, getattr(self, attr),
                                                                                         getattr(other_sim_result,
                                                                                                 attr)))

        self.t_f = max(self.t_f, other_sim_result.t_f)
        self.t_0 = min(self.t_0, other_sim_result.t_0)
        self.finite_elements += other_sim_result.finite_elements

        self.insert_data('x', value=other_sim_result.data['x']['values'], time=other_sim_result.data['x']['time'])
        self.insert_data('y', value=other_sim_result.data['y']['values'], time=other_sim_result.data['y']['time'])
        self.insert_data('u', value=other_sim_result.data['u']['values'], time=other_sim_result.data['u']['time'])

        self.sort()