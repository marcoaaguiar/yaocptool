# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 13:54:58 2016

@author: marco
"""

import sys
#sys.path.append(r"C:\casadi-py27-np1.9.1-v3.0.0")
#sys.path.append(r"C:\coinhsl-win32-openblas-2014.01.10")
#if not 'casadi' in sys.modules:
from casadi import SX,DM, inf, repmat, vertcat, collocation_points, DM, \
                    substitute, cos, sin, pi, diag, horzcat, jacobian, hessian, \
                    sqrt
import matplotlib.pyplot as plt

from yaocptool.modelling.model_classes import SystemModel, SuperModel
from yaocptool.modelling.ocp import OptimalControlProblem, SuperOCP
from yaocptool.modelling.node import Node
from yaocptool.modelling.network import Network

class Tank1(SystemModel):
    def __init__(self, **kwargs):
        
        self.name = 'tank1'        
        SystemModel.__init__(self, Nx =1, Ny= 0, Nz =1, Nu=1)
        
        x_sym = self.x_sym
        z_sym = self.z_sym
        u = self.u_sym
        
        for (k, v) in kwargs.items():
            exec(k + ' = ' + repr(v))
        
        h = x_sym[0]
        q_out = z_sym[0]
        
        q_in = 10
        
        self.ode = vertcat(
            q_in - q_out
        )
        self.alg_z = vertcat(
#            q_out - 0.1*sqrt(h)*u        
            q_out - u        
        )

class Tank2(SystemModel):
    def __init__(self, **kwargs):
        self.name = 'tank2'
        
        SystemModel.__init__(self, Nx =1, Ny= 0, Nz =1, Nu=1)
#        SystemModel.__init__(self, Nx =1, Ny= 0, Nz =1, Nu=0)
        
        x_sym = self.x_sym
        z_sym = self.z_sym
        u = self.u_sym
        
        for (k, v) in kwargs.items():
            exec(k + ' = ' + repr(v))
        
        h = x_sym
        q_out = u
#        q_out = 10
        
        q_in = z_sym
        
        self.ode = vertcat(
            q_in - q_out
        )

class TwoTanks(SuperModel):
    def __init__(self, **kwargs):
        tank1 = Tank1()
        tank2 = Tank2()
        connecting_eq = vertcat(tank1.z_sym - tank2.z_sym)

        SuperModel.__init__(self, [tank1,tank2], connecting_eq)

class StabilizationTank1(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1]), 'R':.1, 'Qv':diag([100]), 'x_ref':DM([2]), 'u_ref':DM([10])}
#        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False        
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5.
        self.x_0 = [1]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2
            
class StabilizationTank2(OptimalControlProblem):
    def __init__(self, model, **kwargs):
        self.cost = {'Q': diag([1]), 'R':.1, 'Qv':diag([100]), 'x_ref':DM([4]), 'u_ref':DM([10])}
#        self.cost = {'Q': diag([1,1,1,1]), 'R':1, 'Qv':diag([1,1,1,1]), 'x_ref':DM([0,0,0,0])}  
        self.state_constraints = False
        self.state_constraints_2 = False
        self.control_constraints = False        
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)
        
        self.t_f  = 5.
        self.x_0 = [1]

        for (k, v) in kwargs.items():
            setattr(self, k, v)
            
        if self.state_constraints:
            self.x_max[2] = 2
            self.x_min[2] = -2
            
class StabilizationTwoTanks(SuperOCP):
    def __init__(self, **kwargs):
        
        self.t_f  = 5.
        self.x_0 = [1,1]      

        tank1 = Tank1()
        tank2 = Tank2()
        
        ocpTank1 = StabilizationTank1(tank1)
        ocpTank2 = StabilizationTank2(tank2)
        
        connecting_eq = vertcat(ocpTank1.model.z_sym - ocpTank2.model.z_sym)
        SuperOCP.__init__(self, problems= [ocpTank1,ocpTank2], **kwargs)
        
        self.model.include_connecting_equations(con = connecting_eq, con_z = ocpTank2.model.z_sym)
#        self.h_final = vertcat(self.h_final, self.model.x_sym[0] -2,self.model.x_sym[1] -4)


class StabilizationTwoTanksCentralized(OptimalControlProblem):
    def __init__(self, **kwargs):
        
        self.t_f  = 5.
        self.x_0 = [1,1]      

        model = TwoTanks()
        
        self.cost = {'Q': diag([1,1]), 'R':diag([.1,.1]), 'Qv':diag([0]), 'x_ref':DM([2,2])}  
        
        OptimalControlProblem.__init__(self, model, obj = self.cost)

def createTwoTanksNetwork():
    tank1 = Tank1()
    tank2 = Tank2()
    
    ocpTank1 = StabilizationTank1(tank1)
    ocpTank2 = StabilizationTank2(tank2) 
    
    node1 = Node(node_id = 1, problem = ocpTank1)
    node2 = Node(node_id = 2, problem = ocpTank2)
    
    nodes = [node1, node2]
    connections_settings_dict = {}
    connections_settings_dict[0] = {1:[0], 2:[0]}
    net = Network(nodes, connections_settings_dict)
    return net
    
if __name__ == '__main__':
    sys_model = TwoTanks()
    
    prob = StabilizationTwoTanks()
    prob2 =StabilizationTwoTanksCentralized()
