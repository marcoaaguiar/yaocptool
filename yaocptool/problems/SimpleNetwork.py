# -*- coding: utf-8 -*-
"""
Created on Fri Jul 07 16:43:42 2017

@author: marco
"""
import sys
from os.path import dirname, abspath
sys.path.append(abspath(dirname(dirname(__file__))))

from casadi import vertcat
from yaocptool.modelling.model_classes import SystemModel
from yaocptool.modelling.ocp import OptimalControlProblem
from yaocptool.modelling.node import Node
from yaocptool.modelling.network import Network



class SimpleNode(Node):
    def __init__(self, node_id = -1, name = '', outputs = 1, outputs_weights=None, control_weight=None, **kwargs):
        if outputs_weights is None:
            outputs_weights = [1]
        if control_weight is None:
            control_weight = []
        for (k, v) in kwargs.items():
            exec(k + ' = ' + repr(v))
            
        if control_weight != 0:
            Nu = 2
        else:
            Nu = 1
        model = SystemModel(Nx = 1, Nz = outputs, Nu= Nu, name = 'node_' + repr(node_id))
        if Nu == 2:
            ode = vertcat(model.u_sym[1]*control_weight + sum([model.z_sym[n_z]*outputs_weights[n_z] for n_z in range(outputs)]))
        else:
            ode = vertcat(sum([model.z_sym[n_z]*outputs_weights[n_z] for n_z in range(outputs)]))
            
        model.include_system_equations(ode = ode,
                                       alg_z = vertcat(model.u_sym[0] - model.z_sym[1])
                                       )
                        
        problem = OptimalControlProblem(model, obj = {'Q':[1], 'R':[10], 'x_ref':[reference]}, x_0 = x_0, t_f =10)
        problem.h_final = vertcat(problem.h_final, problem.model.x_sym - reference)
        
        Node.__init__(self, node_id = node_id, name = name, problem = problem)
    
def createRing(N_nodes):
    nodes = []
    controls = [0]*N_nodes 
    controls[0] = 1
    controls[-1] = -1
    for n in range(N_nodes):
        node = SimpleNode(node_id = n,
                          name = 'node_' + repr(n),
                          outputs = 2,
                          outputs_weights = [1,-1],
                          reference = 5 + n,
                          control_weight = controls[n],
                          x_0 = 10,
                          )
        nodes.append(node)
    
    ## Create Network
    connections_settings_dict = {}
#    connections_settings_dict[0] = {0:[0], N_nodes-1:[1]} 
    for n in range(N_nodes):
        connections_settings_dict[n] = {(n+1)%N_nodes:[0], n%N_nodes:[1]}
    net = Network(nodes, connections_settings_dict)
    return net
