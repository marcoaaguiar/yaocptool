# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

from casadi import DM, diag, sqrt
from yaocptool.modelling import SystemModel, OptimalControlProblem


class Tank1(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="tank_1", model_name_as_prefix=True, **kwargs)

        h = self.create_state('h')
        q_out = self.create_algebraic_variable('q_out')
        u = self.create_control('u')

        q_in = 0.2

        ode = [q_in - q_out]
        alg = [q_out - 0.1 * sqrt(h) * u]

        self.include_system_equations(ode=ode, alg=alg)


class Tank2(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="tank_2", model_name_as_prefix=True, **kwargs)

        h = self.create_state('h')
        q_out = self.create_algebraic_variable('q_out')
        q_in = self.create_control('q_in')
        u = self.create_control('u')

        ode = [q_in - q_out]
        alg = [q_out - 0.1 * sqrt(h) * u]

        self.include_system_equations(ode=ode, alg=alg)


class TwoTanks(SystemModel):
    def __init__(self, **kwargs):
        SystemModel.__init__(self, name="two_tanks", **kwargs)

        tank1 = Tank1()
        tank2 = Tank2()

        self.include_models([tank1, tank2])
        self.connect(tank2.u[0], tank1.y[0])


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
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1, 1]), 'R': diag([.1]), 'Qv': diag([0]), 'x_ref': DM([.5, .5])}
        OptimalControlProblem.__init__(self, model, obj=self.cost, **kwargs)

        self.t_f = 5.
        self.x_0 = [1, 1]

        self.x_min = [0.0001, 0.0001]
        self.u_max = [1, 1]
        self.u_min = [.01, .011]
