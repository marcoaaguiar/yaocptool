"""
This example show different ways to create models in YAOCPTool and how to compose models to create a more complex model.

"""
from casadi import sqrt

from yaocptool.modelling import SystemModel


class Pump(SystemModel):
    def __init__(self, index, **kwargs):
        # define default values
        self.gamma = 0.5
        self.k = 3.33e-6

        # super class method
        SystemModel.__init__(self, name="pump_" + str(index), model_name_as_prefix=True, **kwargs)

        # create variables
        v = self.create_control('v')  # Create input pump voltage
        q = self.create_algebraic_variable('q', 2)  # Create pump flow

        # create equations:
        alg = [q[0] - self.gamma * self.k * v,
               q[1] - (1 - self.gamma) * self.k * v]
        self.include_system_equations(alg=alg)


class Tank(SystemModel):
    def __init__(self, index, **kwargs):
        # define default values
        self.g = 9.8
        self.A = 28e-4
        self.a = 0.071e-4

        # super class method
        SystemModel.__init__(self, name="tank_" + str(index), model_name_as_prefix=True, **kwargs)

        # create variables
        h = self.create_state('h')
        q_in = self.create_control('q_in')
        q_out = self.create_algebraic_variable('q_out')

        ode, alg = self.equations(h, q_in, q_out)

        # create equations:
        self.include_system_equations(ode, alg)

    def equations(self, h, q_in, q_out):
        ode = [q_in - q_out]
        alg = [q_out - self.a * sqrt(2 * self.g * h)]
        return ode, alg


class QuadTanks(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, **kwargs)

        # create the pumps
        pumps = [Pump(index=0, gamma=0.7), Pump(index=1, gamma=0.6)]
        # create the tanks
        tanks = [Tank(index=0, n_inputs=2), Tank(index=1, n_inputs=2), Tank(index=2, n_inputs=1),
                 Tank(index=3, n_inputs=1)]

        self.include_models(pumps)
        self.include_models(tanks)

        # connections
        self.connect(tanks[0].u, pumps[0].y[0] + tanks[2].y[0])
        self.connect(tanks[3].u, pumps[1].y[1] + tanks[3].y[0])
        self.connect(tanks[1].u, pumps[1].y[0])
        self.connect(tanks[2].u, pumps[0].y[1])


if __name__ == "__main__":
    # instantiate the model
    quad_tanks = QuadTanks()

    print(quad_tanks)
