# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 22:03:54 2016

@author: marco
"""
import sys
from os.path import dirname, abspath
if not 'casadi' in sys.modules:
    sys.path.append(abspath(dirname(dirname(__file__))))
    sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
    sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")

import matplotlib.pyplot as plt
from casadi import *

finite_elements = 20

Nx= 6
Ny = 1
Nu =2
x_sym = SX.sym('x', Nx)
#y_sym = self.y_sym 
u_sym = SX.sym('u',Nu)
#y_sym = SX.sym('y', Ny)
u = u_sym[0]
v = u_sym[1]

#model extracte from Tracking trajectories of the cart-pendulum system
# Mazenc

g =  9.8
l = 9.8/9.
M = 1.
m = 1.
eq_max = 2
eq_min = -2

mu = 1e0

#        m = 0.853
#        M = 1
#        l = 0.323

theta = x_sym[0]
theta_dot = x_sym[1]

x = x_sym[2]
x_dot = x_sym[3]

xi = x_sym[4]
dxi = x_sym[5]
psi = eq_max - (eq_max-eq_min)/(1+exp(4*xi/(eq_max-eq_min)))

L =  mtimes(mtimes(x_sym[:4].T,diag([10,0.1,0.1,0.1])), x_sym[:4]) + .1*u**2 + .1*v**2 
Final_V = mtimes(mtimes(x_sym[:4].T,diag([1,1,1,.1])),x_sym[:4])
d_Final_V = gradient(Final_V, x_sym)
d_Final_V_funct = Function('final_v', [x_sym],[d_Final_V])

ode = vertcat(
    theta_dot,
    ((m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2)*cos(theta) +g*sin(theta))/l,
    x_dot,
    (m*g*cos(theta)*sin(theta) -m*l*theta_dot**2*sin(theta)-u)/(M+m-m*cos(theta)**2),
    dxi,
    v,
#
#    L
)
L += + mu*(hessian(psi, xi)[0]*dxi**2+ v*gradient(psi, xi) - ode[3])**2
x_0 = DM([pi/6, 0,0,0,0,0])

## Indir
lamb = SX.sym('lamb', Nx)

H = L + dot(lamb, ode)
ode = vertcat(ode,gradient(H, x_sym))
x_0 = vertcat(x_0, DM.zeros(Nx))
ddH_dudu, dH_du = hessian(H,u_sym)
u_opt = -mtimes(inv(ddH_dudu),substitute(dH_du, u_sym,0))
ode = substitute(ode, u_sym, u_opt)
#alg = vertcat(hessian(psi, xi)[0]*dxi**2/gradient(psi, xi)+ v - ode[3]/gradient(psi, xi))

#system = {'x':x_sym, 'ode':ode, 'p': u_sym, 'z': y_sym, 'alg': alg}
system = {'x':vertcat(x_sym,lamb), 'ode':ode}#, 'p': u_sym}

options = {'t0': 0., 'tf':5./finite_elements}
I = integrator("I", "cvodes", system, options)       

X = []

#for i in range(20):
#    x_f = I(x0 = x_0)['xf']
#    X.append(x_f)
#    x_0 = x_f

## MS
X = MX.sym('var', 2*Nx, finite_elements+1)
U = MX.sym('var', Nu, finite_elements)

if 'Vsol' in locals():
    init_guess = Vsol['x']
else:
    init_guess = vertcat(vec(repmat(x_0, 21)), vec(repmat(DM.zeros(Nu), 20)))
    
G = [X[:Nx,0] - x_0[:Nx]]
for i in range(finite_elements):
    x_f = I(x0 = X[:,i])['xf'] #, p = U[:,i]
    G.append(X[:,i+1] - x_f)
    x_0 = x_f
G.append(X[Nx:,-1]- d_Final_V_funct(X[:Nx,-1]))
G = vertcat(*G)
V = vertcat( vec(X), vec(U))


solver = nlpsol('solver', 'ipopt', {'x':V, 
                                    'f': 0,#X[-1,-1], #+ mtimes(mtimes(X[:4,-1].T,diag([100,100,1,1])),X[:4,-1]), 
                                    'g':G})

Vsol = solver(lbg = 0, ubg = 0, x0 = init_guess)

x = Vsol['x'][:(finite_elements+1)*2*Nx]
x = [x[Nx*2*k: Nx*2*(k+1)] for k in range(finite_elements+1)]
#u = Vsol['x'][21*Nx:]
#u =[u[Nu*k: Nu*(k+1)] for k in range(finite_elements)]


plot_x = [0,2,4]
for ind in plot_x:
    plt.plot(horzcat(*x)[ind, :].T)









 
