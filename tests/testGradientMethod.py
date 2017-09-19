# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:42:23 2016

@author: marco
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from yaocptool.problems.cartpendulum import *
from yaocptool.methods import GradientMethod

model = PendulumCart() 

problem = UpwardPendulumStabilization(model)
#prob.x_max[2] = 5
#prob.x_min[2] = -5


grad = GradientMethod(problem, max_iter =1, degree = 1, finite_elements = 80, integrator_type = 'implicit')
x_sol, u_sol, V_sol = grad.solve()
x, u, t= grad.plotSimulate(x_sol, u_sol, [{'x':[0]},{'u':[0]}], 1, integrator_type = 'implicit')

U = dict(zip(range(grad.finite_elements), [float(i) for i in u]))
X = dict(zip(range(grad.finite_elements), [i.full() for i in x_sol]))