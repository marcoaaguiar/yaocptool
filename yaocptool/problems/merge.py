# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 11:32:09 2017

@author: marco
"""

from model import SystemModel

def merge_models(models_list, connecting_equations = []):
    system = SystemModel(Nx = 0, Ny = 0, Nu = 0, Nz = 0)
    
    for model in models_list:
        system.include_state(model.x_sym, model.ode, model.x_0_sym)
        system.includeAlgebraic(model.y_sym, model.alg)
        system.includeExternalAlgebraic(model.z_sym, model.alg_z)
        system.includeControl(model.u_sym)
        system.includeParameter(model.p_sym)
        system.includeTheta(model.theta_sym)

    system.includeSystemEquations(con = connecting_equations)
    
    return system