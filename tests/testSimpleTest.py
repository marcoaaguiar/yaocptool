# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Fri Jul 07 16:50:09 2017

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

import yaocptool
from yaocptool.problems.SimpleNetwork import *
from yaocptool.methods import IndirectMethod, DirectMethod, AugmentedLagrangian, SequentialAugmentedLagrange
import time 
#model = TwoTanks() 


net = createRing(3)


t1 = time.time()
solution_method = SequentialAugmentedLagrange(net, IndirectMethod,
                                              {},
        max_iter = 5, mu_0 = 1, beta= 10., finite_elements = 30, degree = 5, integrator_type = 'implicit')

x_sol, u_sol, V_sol = solution_method.solve()
print(time.time() - t1)
x, y, u, t= solution_method.plot_simulate(x_sol, u_sol, [{'x':[0, 2, 4]}, {'u':[0, 1]}], 5, integrator_type ='implicit', time_division ='linear')
