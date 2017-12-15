# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:27:37 2016

@author: marco
"""

SOLVER_OPTIONS = {
    'nlpsol_options': {
        "ipopt.linear_solver": "ma27",
        'ipopt.max_iter': 400,
        # 'ipopt.print_level': 0,
        # 'print_time': False,
        # "ipopt.hessian_approximation": "limited-memory",
        # 'ipopt.tol': 1e-14,
    },
}

INTEGRATOR_OPTIONS = {
    # 'abstol' : 1e-10, # abs. tolerance
    # 'reltol' :  1e-10 # rel. tolerance
}
