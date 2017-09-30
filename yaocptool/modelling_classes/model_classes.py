# -*- coding: utf-8 -*-
"""
Created on Thu Jun 09 10:50:48 2016

@author: marco
"""

# if not 'casadi' in sys.modules:
from yaocptool import config
from casadi import SX, DM, inf, repmat, vertcat, collocation_points, \
    substitute, Function, integrator, mtimes, vec, is_equal

import copy


# TODO: fix PEP 8

class SystemModel:
    def __init__(self, Nx=0, Ny=0, Nz=0, Nu=0, Np=0, Ntheta=0, **kwargs):
        """
            x - states
            y - (internal) algebraic
            z - external algebraic
            u - control
            p - constant paramters
            theta - time dependent parameters (finite element)
            u_par - parametrized control parameters

            Note: when vectorizing the parameters order is [ p; theta; u_par]
        """

        # Number of states
        # Number of (internal) algebraic
        # Number of external algebraic
        # Number of control
        # Number of parameters
        # Number of parameters that depend on the finite element

        self.ode = vertcat([])  # ODE
        self.alg = vertcat([])  # Algebraic equations
        self.alg_z = vertcat([])  # Algebraic equations of the z variable
        self.con = vertcat([])  # Connecting algebraic equations
        self.relaxed_alg = vertcat([])

        for (k, v) in kwargs.items():
            setattr(self, k, v)

        if not hasattr(self, 'name'):
            self.name = ''
        else:
            self.name += '_'

        self.x_sym = SX.sym(self.name + 'x', Nx)
        self.x_0_sym = SX.sym(self.name + 'x_0_sym', Nx)
        self.y_sym = SX.sym(self.name + 'y', Ny)
        self.z_sym = SX.sym(self.name + 'z', Nz)
        self.u_sym = SX.sym(self.name + 'u', Nu)
        self.p_sym = SX.sym(self.name + 'p', Np)
        self.theta_sym = SX.sym(self.name + 'theta', Ntheta)

        self.t_sym = SX.sym('t')
        self.tau_sym = SX.sym('tau')

        self.u_par = vertcat(self.u_sym)
        self.hasAdjointVariables = False
        self.con_z = []

    @property
    def system_type(self):
        if self.Ny + self.Nz > 0:
            return 'dae'
        else:
            return 'ode'

    @property
    def Nx(self):
        return self.x_sym.numel()

    @property
    def Ny(self):
        return self.y_sym.numel()

    @property
    def Nz(self):
        return self.z_sym.numel()

    @property
    def Nu(self):
        return self.u_sym.numel()

    @property
    def Np(self):
        return self.p_sym.numel()

    @property
    def Ntheta(self):
        return self.theta_sym.numel()

    @property
    def all_alg(self):
        return vertcat(self.alg, self.alg_z, self.con)

    @property
    def yz_sym(self):
        return vertcat(self.y_sym, self.z_sym)

    @property
    def Nyz(self):
        return self.Ny + self.Nz

    @property
    def x_sys_sym(self):
        if self.hasAdjointVariables:
            return self.x_sym[:self.Nx / 2]
        else:
            return self.x_sym

    @property
    def lamb_sym(self):
        if self.hasAdjointVariables:
            return self.x_sym[self.Nx / 2:]
        else:
            return SX()
    def __repr__(self):
        s = ''
        s += '='*20 + '\n'
        s += 'Model Name: {}'.format(self.name) + '\n'
        s += 'Number of states (x):         {:4} | Number of algebraic (y):               {:4}'.format(self.Nx,
                                                                                                       self.Ny)
        s += '\n'
        s += 'Number of ext. algebraic (z): {:4} | Number of controls (u):                {:4}'.format(self.Nz,
                                                                                                       self.Nu)
        s += '\n'
        s += 'Number of parameters (p):     {:4} | Number of finite elem. param. (theta): {:4}'.format(self.Np,
                                                                                                       self.Ntheta)
        s += '\n'
        s += '-'*20 + '\n'
        s += 'Number of ODE:                {:4} | Number of algebraic eq:                {:4}'.format(self.ode.numel(), self.alg.numel()) + '\n'
        s += 'Number of external alg. eq.:  {:4} | Number of connecting eq.:              {:4}'.format(self.alg_z.numel(), self.con.numel()) + '\n'
        s += '=' * 20 + '\n'
        return s

    def includeSystemEquations(self, ode=None, alg=None, alg_z=None, con=None):
        if con is None:
            con = []
        if alg_z is None:
            alg_z = []
        if alg is None:
            alg = []
        if ode is None:
            ode = []
        self.ode = vertcat(self.ode, ode)
        self.alg = vertcat(self.alg, alg)
        self.alg_z = vertcat(self.alg_z, alg_z)
        self.con = vertcat(self.con, con)

    def replaceVariable(self, original, replacement, variable_type='other'):
        """
            Replace a variable or parameter by something else.
            Input original and replacement, and also variable type which
            describes which type of variable is being remove to it from the
            counters. Types: 'x', 'y', 'u', 'p', 'ignore'
        """

        self.ode = substitute(self.ode, original, replacement)
        self.alg = substitute(self.alg, original, replacement)
        self.alg_z = substitute(self.alg_z, original, replacement)
        self.con = substitute(self.con, original, replacement)
        self.relaxed_alg = substitute(self.relaxed_alg, original, replacement)

        if variable_type == 'u':
            pass
        elif variable_type == 'other':
            pass
        else:
            raise Exception('Not implemented')

    def includeState(self, var, ode, x_0_sym=None):
        delta_Nx = var.numel()
        self.x_sym = vertcat(self.x_sym, var)
        if x_0_sym is None:
            x_0_sym = SX.sym('x_0_sym', delta_Nx)
        self.x_0_sym = vertcat(self.x_0_sym, x_0_sym)

        self.ode = vertcat(self.ode, ode)
        return x_0_sym

    def includeAlgebraic(self, var, alg):
        self.y_sym = vertcat(self.y_sym, var)
        self.alg = vertcat(self.alg, alg)

    def includeExternalAlgebraic(self, var, alg_z=None):
        if alg_z is None:
            alg_z = []
        self.z_sym = vertcat(self.z_sym, var)
        self.alg_z = vertcat(self.alg_z, alg_z)

    def includeConnectingEquations(self, con, con_z=None):
        if con_z is None:
            con_z = []
        self.con = vertcat(self.con, con)
        self.con_z = vertcat(self.con_z, con_z)

    def includeControl(self, var):
        self.u_sym = vertcat(self.u_sym, var)
        self.u_par = vertcat(self.u_par, var)

    def includeParameter(self, p):
        self.p_sym = vertcat(self.p_sym, p)

    def includeTheta(self, theta):
        self.theta_sym = vertcat(self.theta_sym, theta)

    def removeVariablesFromVector(self, var, vector):
        to_remove = self.findVariablesIndecesInVector(var, vector)
        to_remove.sort(reverse=True)
        for it in to_remove:
            vector.remove([it], [])
        return vector

    def removeAlgebraic(self, var, eq=None):
        self.removeVariablesFromVector(var, self.y_sym)
        if eq is not None:
            self.removeVariablesFromVector(eq, self.alg)

    def removeExternalAlgebraic(self, var, eq=None):
        self.removeVariablesFromVector(var, self.z_sym)
        if eq is not None:
            self.removeVariablesFromVector(eq, self.alg_z)

    def removeConnectingEquations(self, var, eq):
        self.removeVariablesFromVector(var, self.z_sym)
        self.removeVariablesFromVector(eq, self.con)

    def removeControl(self, var):
        to_remove = self.findVariablesIndecesInVector(var, self.u_sym)
        to_remove.sort(reverse=True)

        for it in to_remove:
            self.u_sym.remove([it], [])
            self.u_par.remove([it], [])

    # ==============================================================================
    # TIME
    # ==============================================================================

    def convertFromTimeToTau(self, dae_sys, t_k, t_kp1):
        raise NotImplemented

    def convertExprFromTauToTime(self, expr, t_k, t_kp1):
        t = self.t_sym
        tau = self.tau_sym

        h = t_kp1 - t_k
        return substitute(expr, tau, (t - t_k) / h)

    def convertFromTauToTime(self, dae_sys, t_k, t_kp1):
        dae_sys['ode'] = self.convertExprFromTauToTime(dae_sys['ode'], t_k, t_kp1)
        if 'alg' in dae_sys:
            dae_sys['alg'] = self.convertExprFromTauToTime(dae_sys['alg'], t_k, t_kp1)

    # ==============================================================================
    # MERGE
    # ==============================================================================

    def merge(self, models_list, connecting_equations=[], associated_z=[]):
        if not isinstance(models_list, list):
            models_list = [models_list]

        for model in models_list:
            self.includeState(model.x_sym, model.ode, model.x_0_sym)
            self.includeAlgebraic(model.y_sym, model.alg)
            self.includeExternalAlgebraic(model.z_sym, model.alg_z)
            self.includeControl(model.u_sym)
            self.includeParameter(model.p_sym)
            self.includeTheta(model.theta_sym)

        self.includeConnectingEquations(connecting_equations, associated_z)

    # ==============================================================================
    #   Simulation
    # ==============================================================================
    def getDAESystem(self):
        if self.system_type == 'ode':
            system = {'x': self.x_sym, 'ode': self.ode, 't': self.t_sym}
        else:
            system = {'x': self.x_sym, 'z': vertcat(self.y_sym, self.z_sym), 'ode': self.ode,
                      'alg': vertcat(self.alg, self.alg_z, self.con), 't': self.t_sym}
        if self.Np + self.Ntheta + self.u_par.numel() > 0:
            system['p'] = vertcat(self.p_sym, self.theta_sym, self.u_par)
        return system

    def simulate(self, x_0, t_f, t_0=0, p=None, integrator_type='implicit'):
        dae = self.getDAESystem()

        opts = {'tf': t_f, 't0': t_0}  # final time
        F = self.createIntegrator(dae, opts, integrator_type)
        call = {'x0': x_0, 'p': p}

        return F(**call)['xf']

    def simulateStep(self, x_0, t_0, t_f, p=None, dae_sys=None, integrator_type='implicit'):
        if dae_sys is None:
            dae_sys = self.getDAESystem()

        opts = {'tf': float(t_f), 't0': float(t_0)}  # final time
        F = self.createIntegrator(dae_sys, opts, integrator_type)
        args = {'x0': x_0, 'p': p}

        return F(**args)['xf']

    def simulateInterval(self, x_0, t_0, t_f, t_grid, p=None, dae_sys=None, integrator_type='implicit'):
        if dae_sys is None:
            dae_sys = self.getDAESystem()
        X = []
        Y = []
        for t in t_grid:
            opts = {'tf': float(t), 't0': float(t_0)}  # final time
            F = self.createIntegrator(dae_sys, opts, integrator_type)
            call = {'x0': x_0, 'p': p}

            if p is not None:
                call['p'] = DM(p)

            res = F(**call)
            X.append(res['xf'])
            Y.append(res['zf'])
        return X, Y

    def createIntegrator(self, dae_sys, options, integrator_type='implicit'):
        for k in config.INTEGRATOR_OPTIONS:
            options[k] = config.INTEGRATOR_OPTIONS[k]

        if integrator_type == 'implicit':
            if self.system_type == 'ode':
                I = integrator("I", "cvodes", dae_sys, options)
            else:
                I = integrator("I", "idas", dae_sys, options)
        else:
            if self.system_type == 'ode':
                I = self.createExplicitIntegrator('explicitIntegrator', 'rk4', dae_sys, options)
            else:
                raise Exception('explicit integrator not implemented')
        return I

    def createExplicitIntegrator(self, name, integrator_type, dae_sys, options={'t0': 0, 'tf': 1}):
        if 'alg' in dae_sys:
            raise Exception('Explicit integrator not implemented for DAE systems')
        f_in = [dae_sys['t'], dae_sys['x']]
        if 'p' in dae_sys:
            f_in.append(dae_sys['p'])
        else:
            f_in.append(SX.sym('fake_p'))
        f = Function('f_ode', f_in, [dae_sys['ode']])

        t_0 = options['t0']
        t_f = options['tf']

        N_states = dae_sys['x'].numel()
        if integrator_type == 'rk4':
            def RungeKutta4thOrder(x0=DM.zeros(N_states, 1), p=[], iterations=4):
                h = (t_f - t_0)/iterations
                t = t_0
                for it in range(iterations):

                    k1 = h * f(t, x0, p)
                    k2 = h * f(t + 0.5 * h, x0 + 0.5 * k1, p)
                    k3 = h * f(t + 0.5 * h, x0 + 0.5 * k2, p)
                    k4 = h * f(t + h, x0 + k3, p)

                    x_f = x0 + 1 / 6. * k1 + 1 / 3. * k2 + 1 / 3. * k3 + 1 / 6. * k4
                    x0 = x_f
                    t += h
                return {'xf': x_f, 'zf': []}

            return RungeKutta4thOrder

    @staticmethod
    def findVariablesIndecesInVector(var, vector):
        index = []
        for j in range(vector.size1()):
            for i in range(var.numel()):
                if is_equal(vector[j], var[i]):
                    index.append(j)
        return index


#######################################################################

class SuperModel(SystemModel):
    def __init__(self, models=[], connections=[], **kwargs):
        self.models = models
        SystemModel.__init__(self, **kwargs)

        connecting_equations, free_zs = zip(*connections)
        self.merge(self.models, connecting_equations=connecting_equations)
