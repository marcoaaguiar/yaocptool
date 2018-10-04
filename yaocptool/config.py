# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:27:37 2016

@author: marco
"""

SOLVER_OPTIONS = {
    'nlpsol_options': {
        # 'ipopt.print_level': 0,
        # 'print_time': False,
        "ipopt.linear_solver": "ma27",
        # 'ipopt.mumps_pivtol': 1e-8,
        # "ipopt.ma27_liw_init_factor": 15,
        # "ipopt.ma27_la_init_factor": 15,
        'ipopt.max_iter': 40000,
        # 'expand': True,
        # 'jit': True,
        # "ipopt.hessian_approximation": "limited-memory",
        # 'ipopt.tol': 1e-14,
        # 'ipopt.constr_viol_tol': 1e-14,
        # 'ipopt.compl_inf_tol': 1e-14,
    },
}

INTEGRATOR_OPTIONS = {
    # 'abstol' : 1e-10, # abs. tolerance
    # 'reltol' :  1e-10 # rel. tolerance
}
