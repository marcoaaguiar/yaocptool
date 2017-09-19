# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 14:32:14 2016

@author: marco
"""


import sys
sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX,DM, inf, repmat, vertcat, collocation_points, DM, dot, mtimes,\
                    substitute, cos, sin, pi, diag, horzcat, jacobian, hessian, integrator
                    
x_sym = SX.sym('x',4)
#y_sym = self.y_sym 
u = SX.sym('u')

#model extracte from Tracking trajectories of the cart-pendulum system
# Mazenc

g =  9.8
l = 9.8/9.
M = 1.
m = 1.

theta = x_sym[0]
theta_dot = x_sym[1]

x = x_sym[2]
x_dot = x_sym[3]

Q = diag([10,0.1,0.1,0.1,0.])
ode = vertcat(
    theta_dot,
    ((m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2)*cos(theta) +g*sin(theta))/l,
    x_dot,
    (m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2),
#    mtimes(mtimes((x_sym).T, Q),(x_sym)) + 0.1*u**2
)


dae_sys = {'x':x_sym, 'p':u, 'ode':ode}



x_0 = DM([pi/6.,0,0,0])
X = [x_0]
U = [DM(22.8555),
 DM(10.7185),
 DM(3.91643),
 DM(-0.360628),
 DM(-2.8285),
 DM(-4.01295),
 DM(-4.41506),
 DM(-4.39463),
 DM(-4.16502),
 DM(-3.84228),
 DM(-3.48721),
 DM(-3.13077),
 DM(-2.7881),
 DM(-2.46603),
 DM(-2.16711),
 DM(-1.89176),
 DM(-1.63943),
 DM(-1.40911),
 DM(-1.19967),
 DM(-1.00994),
 DM(-0.838796),
 DM(-0.685156),
 DM(-0.547995),
 DM(-0.426345),
 DM(-0.31928),
 DM(-0.225926),
 DM(-0.145448),
 DM(-0.0770582),
 DM(-0.020011),
 DM(0.0263889),
 DM(0.0627791),
 DM(0.089722),
 DM(0.107673),
 DM(0.116925),
 DM(0.117509),
 DM(0.109019),
 DM(0.0903263),
 DM(0.0590779),
 DM(0.0108876),
 DM(-0.0619925)]

h = 0.125
for i in range(40):
    integrator_options= {
            'abstol' : 1e-14, # abs. tolerance
            'reltol' :  1e-14, # rel. tolerance
            't0':i*h, 'tf':(i+1)*h
        }
    I = integrator("I", "cvodes", dae_sys, integrator_options)   

    x_f = I(x0 = X[i], p = U[i])['xf']
    X.append(x_f)


XB = [X[-1]]
UB = list(reversed(U))
dae_sys['ode'] = -dae_sys['ode']
for i in range(40):
    integrator_options= {
            'abstol' : 1e-14, # abs. tolerance
            'reltol' :  1e-14, # rel. tolerance
            't0':i*h, 'tf':(i+1)*h
        }
        
    I = integrator("I", "cvodes", dae_sys, integrator_options)   

    x_f = I(x0 = XB[i], p = UB[i])['xf']
    XB.append(x_f)









     
