# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:43:42 2017

@author: marco
"""
from __future__ import division
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from casadi import vertcat, exp, diag
from yaocptool.modelling_classes.model_classes import SystemModel
from yaocptool.modelling_classes.ocp import OptimalControlProblem


def create_CSTR_OCP():
    model = SystemModel(Nx=4, Ny=3, Nu=2, name='cstr')
    x = model.x_sym
    y = model.y_sym
    u = model.u_sym

    c_A = x[0]
    c_B = x[1]
    theta = x[2]
    theta_K = x[3]

    k_1 = y[0]
    k_2 = y[1]
    k_3 = y[2]

    V_dotV_R = u[0]
    Q_K_dot = u[1]

    ### PARAMETERS

    k_10 = (1.287 * 10 ** 12)
    k_20 = (1.287 * 10 ** 12)
    k_30 = (9.043 * 10 ** 9)
    E_1 = -9758.3
    E_2 = -9758.3
    E_3 = -8560.
    DeltaH_R_AB = 4.2
    DeltaH_R_BC = -11.0
    DeltaH_R_AD = -41.85
    rho = 0.9342
    C_p = 3.01
    k_w = 4032.
    A_R = 0.215
    V_R = 0.01
    m_K = 5.0
    C_PK = 2.0

    ### "Disturbances"
    c_A0 = 5.10
    theta_0 = 104.9  # 100 < theta_0 < 115

    ### INITIAL CONDITIONS

    theta_0_init = 104.9

    c_A_init = 2.14
    c_B_init = 1.09
    theta_init = 114.2
    theta_K_init = 112.9

    V_dotV_R_init = 14.19
    Q_K_dot_init = -1113.5

    u_ref = vertcat(V_dotV_R_init, Q_K_dot_init)
    ### EQUATIONS

    # k_1 = k_10 * exp(E_1 / (theta + 273.15))
    # k_2 = k_20 * exp(E_2 / (theta + 273.15))
    # k_3 = k_30 * exp(E_3 / (theta + 273.15))

    ode = vertcat(
        V_dotV_R * (c_A0 - c_A) - k_1 * c_A - k_3 * c_A ** 2,
        ##
        -V_dotV_R * c_B + k_1 * c_A - k_2 * c_B,
        ##
        V_dotV_R * (theta_0 - theta) - 1 / (rho * C_p) * (
            k_1 * c_A * DeltaH_R_AB + k_2 * c_B * DeltaH_R_BC + k_3 * c_A ** 2 * DeltaH_R_AD)
            + k_w * A_R / ( rho * C_p * V_R) * (theta_K - theta),
        ##
        1 / (m_K * C_PK) * (Q_K_dot + k_w * A_R * (theta - theta_K))
    )

    alg = vertcat(k_10 * exp(E_1 / (theta + 273.15)) - k_1,
                  k_20 * exp(E_2 / (theta + 273.15)) - k_2,
                  k_30 * exp(E_3 / (theta + 273.15)) - k_3
                  )

    model.includeSystemEquations(ode=ode, alg=alg)
    # model.includeSystemEquations(ode=ode)

    x_0 = vertcat([c_A_init, c_B_init, theta_init, theta_K_init])
    x_ref = vertcat([2.14, 1., 110., 105])
    problem = OptimalControlProblem(model, obj={'Q': diag([.1, 10, 1e-3, 1e-3]),
                                                'R': diag([1/3000, 1/2.5]),
                                                # 'Qv':diag([0.1, 1, 0, 0]),
                                                'x_ref': x_ref}, x_0=x_0,
                                    t_f=500./3600, positive_objective = True)
    problem.u_ref = u_ref
    print(problem.t_f)
    ### Constraints
    #c_A
    problem.x_min[0] = 0.1

    #c_B
    problem.x_min[1] = 0.8
    problem.x_max[1] = 1.09

    #theta
    problem.x_min[2] = 0.
    problem.x_max[2] = 300.

    #theta_k
    problem.x_min[3] = 0.
    problem.x_max[3] = 300.

    #V_dotV_R
    problem.u_min[0] = 3.
    problem.u_max[0] = 35.

    problem.u_min[1] = -9000.
    problem.u_max[1] = 0.

    print problem.u_max, problem.u_min
    return problem
