# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:21:14 2017

@author: marco
"""

import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

import yaocptool

from casadi import SX,DM, inf, repmat, vertcat, log, \
                    substitute, exp, jacobian, Function, gradient, depends_on, dot, rootfinder, hessian, \
                    integrator
                    
x = SX.sym('x',2)
xi = SX.sym('xi')
lamb = SX.sym('lamb',2)

u = SX.sym('u')
v = SX.sym('v')
nu = SX.sym('nu')



x_1 = x[0]
x_2 = x[1]
lamb_1 = lamb[0]
lamb_2 = lamb[1]

eq_max = 2
eq_min = -2 

psi = eq_max - (eq_max-eq_min)/(1+exp(4*xi/(eq_max-eq_min)))

dpsi_dxi = gradient(psi, xi)
ode = vertcat(
            x_2,
            -x_1-x_2+lamb_2/2, #u = lamb_2/2
            v,
            #
            2*x_1 -lamb_1 + nu,
            2*x_2 + lamb_1 - lamb_2 +nu
        )
        
alg = vertcat(-x_1-x_2+lamb_2/2 -dpsi_dxi*v,
              nu)

dae_sys = {'x':vertcat(x,xi,lamb), 'ode':ode, 'z':vertcat(v,nu), 'alg':alg, 'p':u}

I = integrator("I", "idas", dae_sys,{'abstol':1e-10, 'reltol':1e-10, 'tf':10})

u0 = 0
x0 = [1,0]
lamb0 = [0,0]


ff = Function('ff',[xi],[-x0[1] - psi])
fpsi = Function('fpsi', [xi],[psi])


finder = rootfinder('f', 'newton',ff)

xi0 = finder().values()[0]
x0 = vertcat(x0, xi0, lamb0)

res = I(x0= x0)
xf = res['xf']
xif = xf[-1]