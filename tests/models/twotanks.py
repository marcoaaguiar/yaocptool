# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

# sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
# sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
# if not 'casadi' in sys.modules:
from casadi import vertcat, DM, \
    diag, sqrt
from yaocptool.modelling import SystemModel, OptimalControlProblem


class Tank1(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="tank_1", model_name_as_prefix=True, **kwargs)

        h = self.create_state('h')
        q_out = self.create_algebraic_variable('q_out')
        u = self.create_control('u')

        q_in = 10

        ode = [q_in - q_out]
        alg = [q_out - 0.1 * sqrt(h) * u]

        self.include_system_equations(ode=ode, alg=alg)


class Tank2(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="tank_2", model_name_as_prefix=True, **kwargs)

        h = self.create_state('h')
        q_out = self.create_algebraic_variable('q_out')
        q_in = self.create_control('q_in')

        ode = [q_in - q_out]
        alg = [q_out - 0.1 * sqrt(h)]

        self.include_system_equations(ode=ode, alg=alg)


class TwoTanks(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="two_tanks", **kwargs)

        tank1 = Tank1()
        tank2 = Tank2()

        self.include_models([tank1, tank2])
        self.connect(tank2.u, tank1.y)


class StabilizationTank1(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1]), 'R': .1, 'Qv': diag([100]), 'x_ref': DM([2]), 'u_ref': DM([10])}
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False

        OptimalControlProblem.__init__(self, model, obj=self.cost, **kwargs)

        self.t_f = 5.
        self.x_0 = [1]

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2


class StabilizationTank2(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1]), 'R': .1, 'Qv': diag([100]), 'x_ref': DM([4]), 'u_ref': DM([10])}
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False

        OptimalControlProblem.__init__(self, model, obj=self.cost, **kwargs)

        self.t_f = 5.
        self.x_0 = [1]

        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2


class StabilizationTwoTanks(OptimalControlProblem):
    def __init__(self, **kwargs):
        self.t_f = 5.
        self.x_0 = [1, 1]

        model = TwoTanks()

        self.cost = {'Q': diag([1, 1]), 'R': diag([.1]), 'Qv': diag([0]), 'x_ref': DM([2, 2])}

        OptimalControlProblem.__init__(self, model, obj=self.cost, **kwargs)
