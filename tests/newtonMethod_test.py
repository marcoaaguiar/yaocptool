# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 17:54:58 2017

@author: marco
"""

from casadi import SX, exp, Function, mtimes, jacobian, jtimes



x = SX.sym('x',2)
f = x[0]**2+x[1]**2 -(x[1]+1)*exp(x[0])

F = Function('F', [x],[f])

def newton(f, x):
    alpha = 0.5
    
    dF = jacobian(f,x)
    
    xpp_funct = Function('xpp', [x], [x- alpha*mtimes(dF,F(x))])
    return xpp_funct
 
xpp = newton(f,x)
x0 = [0,0]   
for i in range(10):
    x0 = xpp(x0)