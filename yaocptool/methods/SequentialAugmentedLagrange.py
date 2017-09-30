# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 16:39:52 2017

@author: marco
"""
from casadi import DM, SX, MX, depends_on, dot, vertcat, is_equal, Function, \
                   horzcat, vec, collocation_points
import matplotlib.pyplot as plt

from AugmentedLagrange import AugmentedLagrange

class SequentialAugmentedLagrange(AugmentedLagrange): 
    def __init__(self, network, Ocp_solver_class, solver_options ={}, **kwargs):
        self.network = network
        self.problems = self.network.problems
        self.Nproblems = len(self.problems)
        self.con = self.network.getConnectionEquations()
        self.con_z = self.network.getConnectionDefinedZ()
        self.approximation_data = {}
        self.used_approximation_data = {}
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        AugmentedLagrange.__init__(self, self.problems[0], Ocp_solver_class, solver_options, 
                                   initialize = False, 
                                   relax_algebraic = False, 
                                   relax_external_algebraic = False, 
                                   relax_connecting_equations = False, 
                                   relax_state_bounds = False)
        
        for (k, v) in kwargs.items():
            setattr(self, k, v)
        
        self.mu = self.mu_0
                                
        solver_options['finite_elements'] = self.finite_elements
        # Create structure for constraints
        self.con_dict = {}
        for k in xrange(self.con.numel()):
            self.con_dict[k] ={'con':self.con[k], 
                               'con_z': self.con_z[k],
                               'associated_problems': [], 
                               'associated_z_sym':[], 
                               'nu_sym': vertcat([]),
                               'nu_pol': vertcat([]),
                               'nu_par': vertcat([]),
                               'nu_tau_sym': vertcat([]),
                               'nu_dict': {} 
                                }
        self.Nr = self.con.numel()
        
        # Create structure for problems
        self.problems_dict ={}
        for p in xrange(len(self.problems)):
            self.problems_dict[p] = {'problem': self.problems[p], 
                                     'connections_id':[],
                                     'associated_problems':[],                                               
#                                     'requested_z': vertcat([]),   
                                     'exog_z_sym': [],
                                     'exog_z_par': [], 
                                     'exog_z_pol': [],
                                     'ocp_solver': None,
                                     'nlp_solver': None,
                                     'Nz_free': 0,
                                     'control_id_of_free_z_dict':{}}
                                     
        # Look for the connections_id associated to each problem.
        for p in xrange(len(self.problems)):
            for k in self.con_dict:
                if depends_on(self.con_dict[k]['con'], self.problems_dict[p]['problem'].model.z_sym):
                    self.problems_dict[p]['connections_id'].append(k)
                    self.con_dict[k]['associated_problems'].append(p)
                    # find the z_sym on the connections and save it on the connection dict
                    for n_z in xrange(self.problems_dict[p]['problem'].model.Nz):
                        tested_nz = self.problems_dict[p]['problem'].model.z_sym[n_z]
                        if depends_on(self.con_dict[k]['con'], self.problems_dict[p]['problem'].model.z_sym[n_z]):
                            self.con_dict[k]['associated_z_sym'].append(tested_nz)
        # Create a function for the connection eq
        for k in self.con_dict:
            self.con_dict[k]['con_funct'] = Function('con_'+`k`+'_function', [vertcat(*self.con_dict[k]['associated_z_sym'])],[self.con_dict[k]['con']])
        
        # Find the exogenous z_sym of each problem
        for p in xrange(len(self.problems)):
            for k in self.problems_dict[p]['connections_id']:
                for z in self.con_dict[k]['associated_z_sym']:
                    if not depends_on(self.problems_dict[p]['problem'].model.z_sym, z):
                        self.problems_dict[p]['exog_z_sym'].append(z)

        # Build relation between problems and connections
        for p in self.problems_dict:
            for k in self.con_dict:
                if p in self.con_dict[k]['associated_problems']:
                    self.problems_dict[p]['associated_problems'].extend(self.con_dict[k]['associated_problems'])
                    self.problems_dict[p]['associated_problems'] = list(set(self.problems_dict[p]['associated_problems']))
                    self.problems_dict[p]['associated_problems'].remove(p)
                    
                    
        ## initializaiton        
        self.includeAugmentedLagrangianTermInTheObjective()
        self.transformFreeZInControl()
        self.parametrizeVariablesInProblems()
        self.initializeApproximationData()
        self.initialize_nu_values()
        ## itialized ocp solver        
        for p in self.problems_dict:
            self.problems_dict[p]['ocp_solver'] = Ocp_solver_class(self.problems_dict[p]['problem'], **solver_options)
    
            
    def transformFreeZInControl(self):
        for problem_id in self.problems_dict:
            for k in self.problems_dict[problem_id]['connections_id']:
                for n_z in self.problems_dict[problem_id]['problem'].model.findVariablesIndecesInVector(self.con_dict[k]['con_z'], self.problems_dict[problem_id]['problem'].model.z_sym):
                 #xrange(self.problems_dict[problem_id]['problem'].model.Nz):
                    print problem_id, n_z, self.problems_dict[problem_id]['problem'].model.Nz, self.problems_dict[problem_id]['problem'].model.z_sym
                    tested_z = self.problems_dict[problem_id]['problem'].model.z_sym[n_z]
                    if is_equal(tested_z, self.con_dict[k]['con_z']):
                        index = self.problems_dict[problem_id]['problem'].model.Nu
                        self.problems_dict[problem_id]['problem'].includeControl(tested_z, u_max = self.problem.z_max[n_z], u_min = self.problem.z_min[n_z])
                        self.problems_dict[problem_id]['problem'].removeExternalAlgebraic(tested_z)
                        self.problems_dict[problem_id]['control_id_of_free_z_dict'][index] = tested_z
            
    def includeAugmentedLagrangianTermInTheObjective(self):
        for k in self.con_dict: #for each connecting equation
            con = self.con_dict[k]['con']
            con_z  = self.con_dict[k]['con_z']
            
            self.Nr += 1
            nu_alg = SX.sym('AL_nu_con_'+`k`)
            self.nu_sym = vertcat(self.nu_sym, nu_alg)
            self.con_dict[k]['nu_sym'] = nu_alg
            
            for problem in self.problems: # for all problems
                if depends_on(con, problem.model.z_sym):
                    problem.L += dot(nu_alg, con) + self.mu_sym/2.*dot(con, con)
                if not depends_on(problem.model.p_sym, self.mu_sym):
                    problem.model.includeParameter(self.mu_sym)
    
    def parametrizeVariablesInProblems(self):
        self.parametrizExogenousVariables()
        self.parametrizeNuInProblems()

    def parametrizExogenousVariables(self):
        for p in self.problems_dict:

            for z in self.problems_dict[p]['exog_z_sym']:
                exog_z_pol, exog_z_par = self.createVariablePolynomialApproximation(1, 
                                                                                    self.degree,
                                                                                    z.name() + '_approx',
                                                                                    tau = self.problems_dict[p]['problem'].model.tau_sym)
                                                                                    
                self.problems_dict[p]['problem'].replaceVariable(z, exog_z_pol)
                self.problems_dict[p]['problem'].model.includeTheta(exog_z_par)
                self.problems_dict[p]['exog_z_pol'].append(exog_z_pol)
                self.problems_dict[p]['exog_z_par'].append(exog_z_par)

#==============================================================================
# NU
#==============================================================================
    def parametrizeNuInProblems(self):
        for k in self.con_dict: #for each connecting equation
            con = self.con_dict[k]['con']
            con_z  = self.con_dict[k]['con_z']
            
            nu_alg = self.con_dict[k]['nu_sym']
            nu_pol, nu_par = self.createVariablePolynomialApproximation(1, 
                                                                        self.degree, 'nu_par_'+con_z.name(),
#2                                                                        tau = self.problems_dict[p]['problem'].model.tau_sym
                                                                        )
            self.con_dict[k]['nu_pol'] = nu_pol
            self.con_dict[k]['nu_tau_sym'] = self.model.tau_sym
            self.con_dict[k]['nu_par'] = nu_par
            
            for p in self.con_dict[k]['associated_problems']: # for all problems
                problem = self.problems_dict[p]['problem']
                # parametrize NU
                problem.replaceVariable(nu_alg, nu_pol)
                problem.replaceVariable(self.con_dict[k]['nu_tau_sym'], problem.model.tau_sym)
                problem.model.includeTheta(nu_par)

    
    def initialize_nu_values(self):
        for k in self.con_dict:
            self.con_dict[k]['nu_dict'] = self.create_nu_initial_guess(n_r= 1)
    
    def getProblemNus(self, problem_id):
        problem_nu_dict = {}
        for k in self.problems_dict[problem_id]['connections_id']:
            problem_nu_dict = self.joinThetas(problem_nu_dict, self.con_dict[k]['nu_dict'])
            
        return problem_nu_dict
        
    def _update_nu(self):
        for con_id in self.con_dict:
            nu, error = self.calculateConnectionNewNu(con_id)
            self.con_dict[con_id]['nu_dict'] = nu
            self.con_dict[con_id]['g_error'] = error
            
    def createFiniteElementError(self):
        t_list  = collocation_points(self.degree)
        mx_g_data = MX.sym('g_data', self.degree)
        
        error = t_list[0]*mx_g_data[0]**2
        for i in xrange(self.degree-1):
            dt = t_list[i+1] -t_list[i]
            error += dt*(mx_g_data[i]**2+mx_g_data[i+1**2])/2
        funct = Function('finite_elem_error', [mx_g_data], [error])
        return funct
        
    def evaluateFiniteElementError(self, con_id, g_data):
        if not 'error_funct' in self.con_dict[con_id]:
            self.con_dict[con_id]['error_funct'] = self.createFiniteElementError()

        return self.con_dict[con_id]['error_funct'](g_data)      
        
    def calculateConnectionNewNu(self, con_id):
        z_data = self.getNuVariablesData(con_id)
        nu_data = self.con_dict[con_id]['nu_dict']
        con_funct = self.con_dict[con_id]['con_funct']

        key1, key2 = z_data.keys()
        new_nu = {}
        error = 0
        
        for i in nu_data:
            z_data_i_1 = z_data[key1][i]
            z_data_i_2 = z_data[key2][i]
            nu_data_i = self.unvec(nu_data[i])
            g_data_i = horzcat()
            for t in xrange(self.degree):
                g_data_i = horzcat(g_data_i,con_funct(vertcat(z_data_i_1[t],z_data_i_2[t])))
            new_nu[i] = vec(nu_data_i + self.mu*g_data_i)
            error += self.evaluateFiniteElementError(con_id, g_data_i)
        print 'errooooooooooooooooooooooooooooo', error
        return new_nu, error
        
#==============================================================================
# APPROXIMATIONS DATA HANDLING
#==============================================================================
        
    def getProblemExogenousVariablesData(self, problem_id):
        data = dict(zip(range(self.finite_elements), [[]]*self.finite_elements))        
        for z in self.problems_dict[problem_id]['exog_z_sym']:
            z_data = self.approximation_data[z]
            data = self.joinThetas(data, z_data)
        return data
        
    def getNuVariablesData(self, con_id):
        data = {}
        for z in self.con_dict[con_id]['associated_z_sym']:
            data[z] = self.approximation_data[z]
        return data

    def saveApproximationData(self, z_sym, data):
        for key in self.approximation_data:
            if is_equal(key, z_sym):
                self.approximation_data[key] = data

    def initializeApproximationData(self):
        self.approximation_data = {}
        for k in self.con_dict:
            for z_sym in self.con_dict[k]['associated_z_sym']:
                self.approximation_data[z_sym] = self.createConstantTheta(constant = 0, 
                                                                          dimension = 1, 
                                                                          degree = self.degree, 
                                                                          finite_elements = self.finite_elements)
    def saveUsedApproximationData(self, problem_id, approximation_data):
        self.used_approximation_data[problem_id] = approximation_data
        
    def generateProblemApproximationData(self, problem_id, X, U, p =[], theta = {}):
        micro_X, micro_Y, micro_U, micro_t = \
            self.problems_dict[problem_id]['ocp_solver'].simulate(X, U, p = p, 
                                                theta = theta, sub_elements = self.degree, 
                                                time_division = 'radau')

        # For all z defined by the system equations
        for n_z in xrange(self.problems_dict[problem_id]['problem'].model.Nz):
            index = self.problems_dict[problem_id]['problem'].model.Ny + n_z
#            
#            plt.plot(micro_t[1:],
#                 vertcat(*[micro_Y[i][index] for i in range(len(micro_Y))]).full()
#                 )
                 
            z = self.problems_dict[problem_id]['problem'].model.z_sym[n_z]
            micro_z = [micro_Y[k][index] for k in xrange(len(micro_Y))]    
            data= {}
            for i in xrange(self.finite_elements):
                data[i] = vertcat(*micro_z[i*self.degree:(i+1)*self.degree])
            self.saveApproximationData(z, data)      
            
        # and for the free z
        for n_z in self.problems_dict[problem_id]['control_id_of_free_z_dict']:
            z = self.problems_dict[problem_id]['control_id_of_free_z_dict'][n_z]
#            plt.plot(micro_t[1:],
#                 vertcat(*[micro_U[i][1] for i in range(len(micro_U))]).full()
#                 )
            micro_z = [micro_U[k][n_z] for k in xrange(len(micro_U))]    
            data= {}
            for i in xrange(self.finite_elements):
                data[i] = vertcat(*micro_z[i*self.degree:(i+1)*self.degree])
            self.saveApproximationData(z, data)
        plt.show()
        print 'end generate problem'
#==============================================================================
# SOLVER     
#==============================================================================

    def initializeProblemsSolvers(self, x_0):
        if len(DM(x_0).full())>0:
            initial_condition_as_parameter = True
        else: 
            initial_condition_as_parameter = False
            
        for p in self.problems_dict:
            self.problems_dict[p]['nlp_solver'] = self.problems_dict[p]['ocp_solver'].get_solver(initial_condition_as_parameter = initial_condition_as_parameter)
    
    def solveIteration(self, initial_guess, p, theta, x_0, tol = 1e-4, max_iter = 2):
        V_sol = initial_guess
        for it in xrange(max_iter):
            for problem_id in self.problems_dict:
                solver = self.problems_dict[problem_id]['nlp_solver']
                
                exogenous_data = self.getProblemExogenousVariablesData(problem_id)
                self.saveUsedApproximationData(problem_id, exogenous_data)
                problem_nu_dict = self.getProblemNus(problem_id)
                
                theta_k = self.joinThetas(theta, exogenous_data, problem_nu_dict)
                V_sol[problem_id] = solver(V_sol[problem_id], p = vertcat(p,self.mu), theta = theta_k, x_0 = x_0)
                X, U = self.splitXandU(V_sol[problem_id])
                self.generateProblemApproximationData(problem_id, X, U, p = vertcat(p,self.mu), theta = theta_k)
                
        return V_sol
        
    def solve_raw(self, initial_guess = None,  p=[], theta = {}, x_0 = []):
        if not self.solver_initialized is None:
            self.initializeProblemsSolvers(x_0)

        if initial_guess is None:
            V_sol = dict(zip(self.problems_dict.keys(), [None]*self.Nproblems))
            
        it = 0
        while True:
            V_sol = self.solveIteration(V_sol, p = p, theta = theta, x_0 = x_0)
            it+=1

            if it==self.max_iter:
                break
            else:
                self._update_nu()
                self.mu = min(self.mu_max,self.mu*self.beta)

#        print 'Solution time: ', time.time()-t1
        return V_sol
            
    def solve(self, initial_guess = None, p = [], theta = {}):
        V_sol = self.solve_raw(initial_guess, p, theta)
        
        X = {}
        U = {}
        
        for problem_id in V_sol:
            X[problem_id], U[problem_id] = self.splitXandU(V_sol[problem_id])
        return X, U, V_sol
        
        
#==============================================================================
# PLOT 
#==============================================================================
    def simulate(self, X, u, sub_elements = 5, t_0 = None, t_f =None, p = [], theta = {}, integrator_type ='implicit', time_division ='linear'):
        one_problem_id = self.problems_dict.keys()[0]
        all_micro_X = [vertcat([])]*(sub_elements * len(u[one_problem_id]) + 1)
        all_micro_Y = [vertcat([])]*(sub_elements * len(u[one_problem_id]))
        all_micro_U = [vertcat([])]*(sub_elements * len(u[one_problem_id]))
        all_micro_t = [vertcat([])]*(sub_elements * len(u[one_problem_id]) + 1)
        for problem_id in self.problems_dict:
            problem_p = vertcat(p,self.mu)
            problem_theta = self.joinThetas(theta, self.used_approximation_data[problem_id], self.getProblemNus(problem_id))
        
            micro_X, micro_Y, micro_U, micro_t = self.problems_dict[problem_id]['ocp_solver'].simulate(X[problem_id], u[problem_id], sub_elements = sub_elements, t_0 = t_0, t_f =t_f, p = problem_p, theta = problem_theta, integrator_type = integrator_type, time_division = time_division)
            for i in xrange(len(micro_X)):
                all_micro_X[i] = vertcat(all_micro_X[i], micro_X[i])
            for i in xrange(len(micro_Y)):
                all_micro_Y[i] = vertcat(all_micro_Y[i], micro_Y[i])
                all_micro_U[i] = vertcat(all_micro_U[i], micro_U[i])
            all_micro_t = micro_t
        return all_micro_X, all_micro_Y, all_micro_U, all_micro_t
        
#    def plotSimulate(self, X, U, plot_list, n_of_sub_elements =  5, p = [], theta = {}, integrator_type = None):
#        for problem_id in X:
#            if integrator_type == None:
#                integrator_type = self.integrator_type
#            problem_p = vertcat(p,self.mu)
#            problem_theta = self.joinThetas(theta, self.getProblemNus(problem_id), self.getProblemExogenousVariablesData(problem_id))
#            
#            micro_X, micro_Y, micro_U, micro_t =self.problems_dict[p]['ocp_solver'].simulate(X[problem_id], 
#                                U[problem_id], n_of_sub_elements, 
#                                p = problem_p, theta = problem_theta, 
#                                integrator_type = integrator_type)
    
            
            
            
            
                    
                    
