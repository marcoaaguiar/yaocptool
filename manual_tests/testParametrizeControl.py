# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""

from manual_tests.models.vanderpol import VanDerPol

model = VanDerPol()
k = model.create_parameter("k", 2)

model.parametrize_control(model.u, k.T @ model.x)

# sim_result = model.simulate([0, 1], t_f=[0.25 * i for i in range(1, 101)], t_0=0, p=[-1, -1])
sim_result = model.simulate([0, 1], t_f=[0.25 * i for i in range(1, 101)], t_0=0, u=0.1)
sim_result.plot([{"x": "all"}, {"u": "all"}])
