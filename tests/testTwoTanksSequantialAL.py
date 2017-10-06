# -*- coding: utf-8 -*-
from __future__ import print_function

"""
Created on Wed Nov 02 18:57:40 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

import yaocptool
from yaocptool.problems.twotanks import *
from yaocptool.methods import IndirectMethod, DirectMethod, AugmentedLagrangian, SequentialAugmentedLagrange
import time 
#model = TwoTanks() 

problem = StabilizationTwoTanks()


tank1 = Tank1()
tank2 = Tank2() 

ocpTank1 = StabilizationTank1(tank1)
ocpTank2 = StabilizationTank2(tank2)    

con = vertcat(ocpTank1.model.z_sym - ocpTank2.model.z_sym)
con_z = vertcat(ocpTank2.model.z_sym)

problems = [ocpTank1, ocpTank2]

net = createTwoTanksNetwork()
#problem = StabilizationTwoTanksCentralized()
#prob.x_max[2] = 5
#prob.x_min[2] = -5


#solution_method = IndirectMethod(problem, degree = 1, finite_elements = 20, integrator_type = 'implicit')
#solution_method = DirectMethod(problem, degree = 1, finite_elements = 20, integrator_type = 'implicit')
##
t1 = time.time()
solution_method = SequentialAugmentedLagrange(net, IndirectMethod,
                                              {},
        max_iter = 5, mu_0 = 1, beta= 10., finite_elements = 30, degree = 5, integrator_type = 'implicit')

#V_sol = solution_method.solve_raw()

x_sol, u_sol, V_sol = solution_method.solve()
print(time.time() - t1)
x, y, u, t= solution_method.plot_simulate(x_sol, u_sol, [{'x':[0, 2]}, {'u':[0, 1]}], 5, integrator_type ='implicit', time_division ='linear')


#U = dict(zip(range(indir_method.finite_elements), [float(i) for i in u]))
#X = dict(zip(range(indir_method.finite_elements), [i.full() for i in x_sol]))

