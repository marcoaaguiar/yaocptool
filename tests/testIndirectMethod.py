# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import IndirectMethod

model = PendulumCart() 

problem = UpwardPendulumStabilization(model, t_f = 3.)
#prob.x_max[2] = 5
#prob.x_min[2] = -5


indir_method = IndirectMethod(problem, degree = 5, finite_elements = 20, integrator_type = 'implicit')
x_sol, u_sol, V_sol = indir_method.solve()
x, y, u, t= indir_method.plot_simulate(x_sol, u_sol, [{'x':[0, 1]}, {'x':[2, 3]}, {'u':[0]}], 5, integrator_type ='implicit')

U = dict(zip(range(indir_method.finite_elements), [float(i) for i in u]))
X = dict(zip(range(indir_method.finite_elements), [i.full() for i in x_sol]))