# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

from casadi import vertcat, DM, cos, sin, pi, diag

from yaocptool.modelling.ocp import OptimalControlProblem
from yaocptool.modelling.system_model import SystemModel


class PendulumCart(SystemModel):
    def __init__(self, **kwargs):
        super().__init__(self)

        x = self.create_state('x', 4)
        u = self.create_control('u')

        # model extracte from Tracking trajectories of the cart-pendulum system
        # Mazenc

        g = 9.8
        ell = 9.8 / 9.0
        big_m = 1.0
        m = 1.0

        for (k, v) in kwargs.items():
            exec(k + " = " + repr(v))  # comentario
        #        m = 0.853
        #        M = 1
        #        l = 0.323

        theta = x[0]
        theta_dot = x[1]

        x = x[2]
        x_dot = x[3]

        ode = vertcat(
            theta_dot,
            ((m * g * cos(theta) * sin(theta) -
              m * ell * theta_dot**2 * sin(theta) - u) /
             (big_m + m - m * cos(theta)**2) * cos(theta) + g * sin(theta)) /
            ell,
            x_dot,
            (m * g * cos(theta) * sin(theta) -
             m * ell * theta_dot**2 * sin(theta) - u) /
            (big_m + m - m * cos(theta)**2),
        )
        self.include_equations(ode=ode)


class UpwardPendulumStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {
            "Q": diag([10, 0.1, 0.1, 0.1]),
            "R": 0.1,
            "Qv": diag([1, 1, 1, 1]),
            "x_ref": DM([0, 0, 0, 0]),
        }
        #        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False

        OptimalControlProblem.__init__(self, model, obj=self.cost)

        self.t_f = 5.0
        self.x_0 = [pi / 6.0, 0, 0, 0]

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2
        if self.state_constraints_2:
            self.x_max[3] = 3
            self.x_min[3] = -3
        if self.control_constraints:
            self.u_max[0] = 20.0
            self.u_min[0] = -20.0


class DownwardPendulumStabilization(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {
            "Q": diag([10, 0.1, 0.1, 0.1]),
            "R": 0.1,
            "Qv": diag([100, 100, 0, 0]),
            "x_ref": DM([pi, 0, 0, 0]),
        }
        self.state_constraints = False
        self.control_constraints = False

        OptimalControlProblem.__init__(self, model, obj=self.cost)

        self.t_f = 5
        self.x_0 = [pi / 6, 0, 0, 0]

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if self.state_constraints:
            self.x_max[2] = 10
            self.x_min[2] = -10
        if self.control_constraints:
            self.u_max[0] = 2
            self.u_min[0] = -2


if __name__ == "__main__":
    sys_model = PendulumCart()

    prob = UpwardPendulumStabilization(sys_model)
