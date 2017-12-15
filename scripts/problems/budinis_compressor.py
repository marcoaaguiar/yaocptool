#!/usr/bin/env python
""" Implements a compressor model from Budinis (2015)

Compressor model extracted from "Control of centrifugal compressors via model predictive control for enhanced oil
recovery applications" -- S. Budinis. N. F. Thornhill
DOI: 10.1016/j.ifacol.2015.08.002
"""
from casadi import sqrt, diag, pi, fmax, SX, Function, rootfinder

from yaocptool.modelling import SystemModel, OptimalControlProblem


# noinspection PyPep8Naming
def create_compressor():
    x_names = (('p', 1), ('m', 1), ('omega', 1), ('p_01', 1))
    model = SystemModel(name='compressor', x_names=x_names, n_y=5, n_u=2)

    p = model.x_sym[0]  # Plenum pressure
    m = model.x_sym[1]  # Mass flow rate that enters the compressor
    omega = model.x_sym[2]  # Shaft rotational velocity
    p_01 = model.x_sym[3]  # Inlet pressure

    tau_c = model.y_sym[0]
    m_in = model.y_sym[1]
    m_out = model.y_sym[2]
    m_r = model.y_sym[3]
    Psi_c = model.y_sym[4]

    tau_d = model.u_sym[0]
    asv = model.u_sym[1]

    # Paper Parameters
    # a2_01V = (0.001 + 0.005) / 2
    # A_1L = (0.001 + 0.005) / 2
    # J = ((0.5 + 2) / 2) ** (-1)
    # ur2_2 = (0.01 + 0.05) / 2
    # k_in = (1 + 2.5) / 2
    # k_out = (1 + 2.5) / 2
    # k_r = (1 + 2.5) / 2

    a2_01V = 0.3099
    A_1L = 0.02
    J = 150.
    ur2_2 = 0.1063
    # k_in = (1 + 2.5) / 2
    k_in = 3.
    k_out = (1 + 2.5) / 2
    k_r = (1 + 2.5) / 2

    #
    p_factor = 1e5

    # Gas parameters

    R = 8.3144598 * 1e-5
    M_g = 0.019  # kg/mol
    T = 298.15
    T_in = 403.

    # Estimated parameters

    p_in = 6.2
    p_out = 20.
    p_r = p
    m_pout = m_r + m_out

    rho_in = p_in * M_g / (R * T_in)
    rho = p * M_g / (R * T)
    rho_r = p_r *  M_g / (R * T)

    # Compressor map

    Rref = 14000. / 60. * 2. * pi  # Maximum compressor speed
    R_Per = omega / Rref
    F_vol = 60. * (m * R * T / (M_g * p_01))
    x_map = F_vol / R_Per
    y_map = R_Per

    # Psi_c = 1.752 + 0.08592 * x_map - 14.18 * y_map - 0.0004272 * x_map ** 2 + 0.003541 * x_map * y_map + 11.65 * y_map ** 2
    # Psi_c = 1.752 + 0.08592 * x_map - 14.18 * y_map - 0.0004272 * x_map ** 2 + 0.003541 * x_map * y_map + 11.65 * y_map ** 2

    # Debug
    # _m_out = _m = _m_in = 5.
    #
    # _rho_in = p_in * M_g / (R * T_in)
    # _p_01 = p_in - 1 / _rho_in * (_m_in / k_in) ** 2
    # print(k_in * sqrt(_rho_in * (p_in - _p_01)))
    #
    # a = M_g / (R * T)
    # b = -M_g / (R * T) * p_out
    # c = -(_m_out / k_out) ** 2
    # x1 = (-b + sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    # x2 = (-b - sqrt(b ** 2 - 4 * a * c)) / (2 * a)
    # _p = x1
    #
    # _rho = _p * M_g / R / T
    # print(k_out * sqrt(_rho * (_p - p_out)))
    #
    # _p_02 = _p
    # _Psi_c = _p_02 / _p_01
    #
    # _omega = SX.sym('_omega')
    # _Rref = 14000. / 60 * 2 * pi  # Maximum compressor speed
    # _R_Per = _omega / _Rref
    # _F_vol = 60 * (_m * R * T / (M_g * _p_01))
    # _x_map = _F_vol / _R_Per
    # _y_map = _R_Per
    #
    # map_Psi_c = 1.752 + 0.08592 * _x_map - 14.18 * _y_map - 0.0004272 * _x_map ** 2 + 0.003541 * _x_map * _y_map + 11.65 * _y_map ** 2
    # f_map = Function('f_map', [_omega], [map_Psi_c - _Psi_c])
    # rf = rootfinder('rf_psi', 'newton', f_map)
    # _omega_rf = rf(Rref)
    #
    # _tau_d = _tau_c = ur2_2 * _omega_rf * _m

    # Equations

    ode = [a2_01V * (m - m_pout),  # p
           A_1L * (Psi_c * p_01 - p),  # m
           1 / J * (tau_d - tau_c),  # omega
           (10 * a2_01V) * (m_in + m_r - m)]  # p_01

    alg = [tau_c - ur2_2 * omega * m,
           m_in - k_in * sqrt(rho_in * fmax(0.0001, p_in - p_01)),
           m_out - k_out * sqrt(rho * fmax(0.0001, (p - p_out))),
           m_r - k_r * sqrt(rho_r * fmax(0.0001, (p_r - p_01))) * asv,
           Psi_c - (
               1.752 + 0.08592 * x_map - 14.18 * y_map - 0.0004272 * x_map ** 2 + 0.003541 * x_map * y_map
               + 11.65 * y_map ** 2)]


    model.include_system_equations(ode=ode, alg=alg)

    x_0 = [20.51906482375495, 5., 1528.51, 5.409883206140351]

    # x_ref = [20, 10, 1000., 5.]

    L = (m_out - 4.) ** 2 + asv**2

    problem = OptimalControlProblem(model, L=L,
                                    # obj={'Q': diag([0, 10, 0, 0]),
                                    #             'R': diag([0.001, 0]),
                                    #             # 'Qv':diag([0.1, 1, 0, 0]),
                                    #             'x_ref': x_ref},
                                    x_0=x_0,
                                    t_f=1., positive_objective=True)

    # problem.x_min[0] = 0.1

    problem.y_min[1] = 0.1
    problem.y_min[2] = 0.1
    problem.y_min[3] = 0.

    problem.u_min[1] = 0.
    problem.u_max[1] = 1.

    problem.u_min[0] = 0.
    problem.u_max[0] = 3000.

    # p = model.x_sym[0]  # Plenum pressure
    # m = model.x_sym[1]  # Mass flow rate that enters the compressor
    # omega = model.x_sym[2]  # Shaft rotational velocity
    # p_01 = model.x_sym[3]  # Inlet pressure

    return model, problem
