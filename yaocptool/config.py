# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:27:37 2016

@author: marco
"""

from typing import Any, Dict

SOLVER_OPTIONS: Dict[str, Any] = {
    "nlpsol_options": {
        #  "ipopt.print_level": 0,
        #  "print_time": False,
        "ipopt.linear_solver": "ma27",
        "expand": True,
        #  "jit": True,
        # 'ipopt.mumps_pivtol': 1e-8,
        # "ipopt.ma27_liw_init_factor": 15,
        # "ipopt.ma27_la_init_factor": 15,
        #  "ipopt.max_iter": 1000,
        # "ipopt.hessian_approximation": "limited-memory",
        #  "ipopt.tol": 1e-14,
        #  "ipopt.constr_viol_tol": 1e-14,
        #  "ipopt.compl_inf_tol": 1e-14,
    },
}

INTEGRATOR_OPTIONS: Dict[str, Any] = {
    # 'abstol': 1e-10,  # abs. tolerance
    # 'reltol': 1e-10  # rel. tolerance
}

PLOT_INTERACTIVE: bool = False
