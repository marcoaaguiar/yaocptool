# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 15:27:37 2016

@author: marco
"""

SOLVER_OPTIONS = { #'nlpsol': "ipopt", \
    'nlpsol_options': {
       # "ipopt.hessian_approximation":"limited-memory",
        "ipopt.linear_solver":"ma27",
        # 'ipopt.tol': 1e-14,
#        'ipopt.mu_strategy':'adaptive',
#        'ipopt.mu_oracle':'loqo',
        'ipopt.max_iter': 1000,
#        'ipopt.nlp_scaling_method': 'none',
#        'ipopt.print_level':0,
#        'verbose':False,
        },
       'verbose':False
    }
    
INTEGRATOR_OPTIONS = {
       # 'abstol' : 1e-10, # abs. tolerance
       # 'reltol' :  1e-10 # rel. tolerance
}