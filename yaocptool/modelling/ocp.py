# -*- coding: utf-8 -*-
"""
Created on Mon Apr 03 11:15:03 2017

@author: marco
"""

import copy

from casadi import DM, repmat, vertcat, substitute, mtimes, is_equal, SX, inf

from yaocptool.modelling import SystemModel

class OptimalControlProblem:
    def __init__(self, model, **kwargs):
        if not hasattr(self, 't_0'):
            self.t_0 = 0.
        if not hasattr(self, 't_f'):
            self.t_f = 1.
        if not hasattr(self, 'x_0'):
            self.x_0 = DM([])

        self.name = ''
        self.model = SystemModel()
        self._model = model  # type: SystemModel
        self.resetWorkingModel()

        self.x_max = repmat(inf, self.model.Nx)
        self.y_max = repmat(inf, self.model.Ny)
        self.z_max = repmat(inf, self.model.Nz)
        self.u_max = repmat(inf, self.model.Nu)

        self.x_min = repmat(-inf, self.model.Nx)
        self.y_min = repmat(-inf, self.model.Ny)
        self.z_min = repmat(-inf, self.model.Nz)
        self.u_min = repmat(-inf, self.model.Nu)

        self.h_initial = self.model.x_sym - self.model.x_0_sym
        self.h_final = vertcat([])
        self.g_ineq = vertcat([])

        self.L = DM(0.)  # type: DM
        self.V = DM(0.)  # type: DM
        self.H = DM(0.)

        self.eta = SX()

        self.parametrized_control = False
        self.positive_objective = False
        self.NULL_OBJ = False

        if 'obj' in kwargs:
            obj_value = kwargs.pop('obj')
            if type(obj_value) == dict:
                for (k, v) in obj_value.items():
                    setattr(self, k, v)
                self.createQuadraticCost(obj_value)

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        self.x_0 = DM(self.x_0)

        # Treat Initialization
        self.check_integrity()

    @property
    def N_h_final(self):
        return self.h_final.size1()

    @property
    def N_eta(self):
        return self.eta.size1()

    @property
    def yz_max(self):
        return vertcat(self.y_max, self.z_max)

    @property
    def yz_min(self):
        return vertcat(self.y_min, self.z_min)

    def check_integrity(self):
        # Check if Objective Function was provided
        if self.L.is_zero() and self.V.is_zero() and not self.NULL_OBJ:
            raise Exception('No objective')
        elif not hasattr(self, 'L'):
            self.L = DM(0.)
        elif not hasattr(self, 'V'):
            self.V = DM(0.)

        # Check if the objective function has the proper size
        if not self.L.numel() == 1:
            raise Exception(
                'Size of dynamic cost (ocp.L) is different from 1, provided size is: {}'.format(self.L.numel()))
        if not self.V.numel() == 1:
            raise Exception(
                'Size of final cost (ocp.V) is different from 1, provided size is: {}'.format(self.L.numel()))
        return True

    def pre_solve_check(self):
        self._fix_types()
        self.check_integrity()

        # Check if the initial condition has the same number of elements of the model
        attributes = ['x_0', 'x_max', 'y_max', 'z_max', 'u_max', 'x_min', 'y_min', 'z_min', 'u_min']
        attr_to_compare = ['Nx', 'Nx', 'Ny', 'Nz', 'Nu', 'Nx', 'Ny', 'Nz', 'Nu']
        for i, attr in enumerate(attributes):
            if not getattr(self, attr).numel() == getattr(self.model, attr_to_compare[i]):
                raise Exception(
                    'The size of the initial guess "self.{}" is not equal to the number of states "model.{}",'
                    + ' {} != {}'.format(attr, attr_to_compare[i], self.x_0.numel(), self.model.Nx))
        return True

    def _fix_types(self):
        self.x_max = vertcat(self.x_max)
        self.y_max = vertcat(self.y_max)
        self.z_max = vertcat(self.z_max)
        self.u_max = vertcat(self.u_max)

        self.x_min = vertcat(self.x_min)
        self.y_min = vertcat(self.y_min)
        self.z_min = vertcat(self.z_min)
        self.u_min = vertcat(self.u_min)

        self.x_0 = vertcat(self.x_0)

    def resetWorkingModel(self):
        self.model = copy.copy(self._model)

    def createCostState(self):
        x_c = SX.sym('x_c')
        if self.positive_objective:
            x_min = 0
        else:
            x_min = -inf

        self.include_state(x_c, self.L, x_0=0, x_min=x_min)
        #        self.include_state(x_c, self.L, x_0 = 0)
        #        self.h_initial = vertcat(self.h_initial, x_c)
        self.model.x_c = x_c
        self.L = DM(0)
        self.V += x_c

    def makeFinalCostFunction(self, p=None):
        raise Exception('To be removed')
        # if p != None:
        #     self.V_function = Function('FinalCost', [self.model.x_sym, p],[self.V])
        # else:
        #     self.V_function = Function('FinalCost', [self.model.x_sym],[self.V])

    def createQuadraticCost(self, par_dict):
        self.L = DM(0)
        self.V = DM(0)
        if 'x_ref' not in par_dict:
            par_dict['x_ref'] = DM.zeros(self.model.Nx)
        if 'u_ref' not in par_dict:
            par_dict['u_ref'] = DM.zeros(self.model.Nu)

        if 'Q' in par_dict:
            Q = par_dict['Q']
            self.L += mtimes(mtimes((self.model.x_sym - par_dict['x_ref']).T, Q),
                             (self.model.x_sym - par_dict['x_ref']))

        if 'R' in par_dict:
            R = par_dict['R']
            self.L += mtimes(mtimes((self.model.u_sym - par_dict['u_ref']).T, R),
                             (self.model.u_sym - par_dict['u_ref']))

        if 'Qv' in par_dict:
            Qv = par_dict['Qv']
            self.V += mtimes(mtimes((self.model.x_sym - par_dict['x_ref']).T, Qv),
                             (self.model.x_sym - par_dict['x_ref']))

        if 'Rv' in par_dict:
            Rv = par_dict['Rv']
            self.V += mtimes(mtimes(self.model.x_sym.T, Rv), self.model.x_sym)

    def merge(self, problems):
        for problem in problems:
            self.L += problem.L
            self.V += problem.V

            self.x_max = vertcat(self.x_max, problem.x_max)
            self.y_max = vertcat(self.y_max, problem.y_max)
            self.z_max = vertcat(self.z_max, problem.z_max)
            self.u_max = vertcat(self.u_max, problem.u_max)

            self.x_min = vertcat(self.x_min, problem.x_min)
            self.y_min = vertcat(self.y_min, problem.y_min)
            self.z_min = vertcat(self.z_min, problem.z_min)
            self.u_min = vertcat(self.u_min, problem.u_min)

            self.h_initial = vertcat(self.h_initial, problem.h_initial)
            self.h_final = vertcat(self.h_final, problem.h_final)
            self.g_ineq = vertcat(self.g_ineq, problem.g_ineq)

            self.model.merge([problem.model])

            # ==============================================================================
            # INCLUDE VARIABLES
            # ==============================================================================

    def include_state(self, var, ode, x_0=None, x_min=None, x_max=None, h_initial=None, x_0_sym=None, suppress=False):
        if x_min is None:
            x_min = -DM.inf(var.numel())
        if x_max is None:
            x_max = DM.inf(var.numel())

        if x_0 is None and h_initial is None and not suppress:
            raise Exception('No intial condition given')

        x_0_sym = self.model.include_state(var, ode, x_0_sym)

        if x_0 is not None:
            self.x_0 = vertcat(self.x_0, x_0)
            h_initial = x_0_sym - var
        else:
            x_0 = DM.zeros(var.shape)
            self.x_0 = vertcat(self.x_0, x_0)

        if h_initial is not None:
            self.h_initial = vertcat(self.h_initial, h_initial)

        self.x_min = vertcat(self.x_min, x_min)
        self.x_max = vertcat(self.x_max, x_max)

    def include_algebraic(self, var, alg, y_min=None, y_max=None):
        if y_min is None:
            y_min = -DM.inf(var.numel())
        if y_max is None:
            y_max = DM.inf(var.numel())

        self.model.include_algebraic(var, alg)
        self.y_min = vertcat(self.y_min, y_min)
        self.y_max = vertcat(self.y_max, y_max)

    def include_control(self, var, u_min=None, u_max=None):
        if u_min is None:
            u_min = -DM.inf(var.numel())
        if u_max is None:
            u_max = DM.inf(var.numel())

        self.model.include_control(var)
        self.u_min = vertcat(self.u_min, u_min)
        self.u_max = vertcat(self.u_max, u_max)

    def remove_algebraic(self, var, eq=None):
        to_remove = self.model.find_variables_indices_in_vector(var, self.model.y_sym)
        to_remove.reverse()

        for it in to_remove:
            self.y_max.remove([it], [])
            self.y_min.remove([it], [])
        self.model.remove_algebraic(var, eq)

    def remove_external_algebraic(self, var, eq=None):
        to_remove = self.model.find_variables_indices_in_vector(var, self.model.z_sym)
        to_remove.reverse()
        for it in to_remove:
            self.z_max.remove([it], [])
            self.z_min.remove([it], [])
        self.model.remove_external_algebraic(var, eq)

    def remove_connecting_equations(self, var, eq):
        self.model.remove_connecting_equations(var=var, eq=eq)

    def remove_control(self, var):
        to_remove = []
        for j in range(self.model.Nu):
            for i in range(var.numel()):
                if is_equal(self.model.u_sym[j], var[i]):
                    to_remove.append(j)

        to_remove.sort(reverse=True)
        for it in to_remove:
            self.u_max.remove([it], [])
            self.u_min.remove([it], [])

        self.model.remove_control(var)

    def replace_variable(self, original, replacement, variable_type='other'):
        self.L = substitute(self.L, original, replacement)
        if variable_type == 'p':
            self.V = substitute(self.V, original, replacement)

        self.model.replace_variable(original, replacement, variable_type)


class SuperOCP(OptimalControlProblem):
    def __init__(self, problems, **kwargs):
        self.problems = problems
        OptimalControlProblem.__init__(self, SystemModel(), NULL_OBJ=True, **kwargs)
        self.merge(problems)
