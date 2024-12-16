from modules.conflict_graph import ConflictGraph
from modules.qaoa import QAOA
from build_qubo import BuildQUBO
from dadk.QUBOSolverCPU import *
from dadk.QUBOSolverDAv4 import *
from dadk.QUBOSolverDAv3 import *
import sys
import os
from scipy.optimize import minimize, differential_evolution
from multiprocessing import Pool


class SolveQUBO:
    '''
    Class to solve QUBO problem. Solver can be either QUBOSolverCPU (DA) or GateQPU (QAOA). Args:
        - conflict_g: conflict graph
        - solver_type: 'QUBOSolverCPU' or 'GateQPU'
        - factor: factor to multiply edge constraint
    '''
    def __init__(self,
                 conflict_g:ConflictGraph,
                 solver_type:str,
                 factor=2.,
                 **kwargs         
                ):
        self.solver_type = solver_type

        if(solver_type=='GateQPU'):
            # build qaoa object 
            tic = time.time()
            self.qaoa = QAOA(conflict_g,n_layers=kwargs['n_layers'],factor=factor)
            self.build_qubo_time = time.time()-tic
            self.best_individual = []

        # if(solver_type=='QUBOSolverCPU'):
        else:
            # build qubo
            qubo = BuildQUBO(conflict_g)
            qubo.create_qubo(factor=factor)
            self.build_qubo_time = qubo.time
            self.QUBO = qubo 

        

        self.time = 0

    def callback(self, xk, convergence=1e-8):
        self.best_individual.append(list(xk))

    def solve_qubo(self,get_solution_list=False,**kwargs):
        '''
        Class method to solve QUBO problem. 
        Args:
            - backend: qiskit backend
        '''
        if(self.solver_type!='GateQPU'):

            args_minimize = {}
            if 'penalty_qubo' in kwargs and self.solver_type!='QUBOSolverCPU':
                args_minimize['penalty_qubo'] = kwargs['penalty_qubo']
                args_minimize['qubo'] = kwargs['qubo']

            else:
                args_minimize['qubo'] = self.QUBO.qubo

        if(self.solver_type=='QUBOSolverCPU'):
            # initialize solver
            args_initilize = {
                'optimization_method':'parallel_tempering',
                                'number_iterations':1000,
                                'number_runs':10,
                                'scaling_bit_precision':32,
                                'auto_tuning':AutoTuning.AUTO_SCALING_AND_SAMPLING,
                                'solver_max_bits':12000}
            if 'args_initilize' in kwargs:
                args_initilize.update(kwargs['args_initilize'])
            solver = QUBOSolverCPU(**args_initilize)
            
            # solve
            try:
                solution_list = solver.minimize(**args_minimize)
                solution = solution_list.get_minimum_energy_solution()
                 
                 # save time
                self.time = solution_list.solver_times.duration_solve.total_seconds()
                self.all_times = {'execution':solution_list.solver_times.duration_execution.total_seconds(),
                                    'solve':solution_list.solver_times.duration_solve.total_seconds(),
                                    'tuning':solution_list.solver_times.duration_tuning.total_seconds(),
                                    'elapsed':solution_list.solver_times.duration_elapsed.total_seconds()}

            except Exception as e:
                print(e)
                solution = None
                self.time = None
                self.all_times = None          
        
        elif(self.solver_type=='DA_v4'):
            solver = QUBOSolverDAv4(
            time_limit_sec=30,
            auto_tuning=AutoTuning.AUTO_SCALING,
            use_access_profile=True,
            scaling_bit_precision=32,
            #offline_request_file= "request.json",
            #offline_response_file = "response.json", 
            access_profile_file='config_da/annealer_v4.prf')

            solution_list = solver.minimize(**args_minimize)
            solution = solution_list.get_minimum_energy_solution()

        elif(self.solver_type=='DA_v3c'):
            print('DA_v3c')
            print(f'Num_group: {kwargs["num_group"]}')
            solver = QUBOSolverDAv3(
            time_limit_sec=kwargs['time_limit_sec'],
            auto_tuning=AutoTuning.AUTO_SCALING,
            use_access_profile=True,
            scaling_bit_precision=32,
            num_group=kwargs['num_group'],
            offline_request_file= "request.json",
            offline_response_file = "response.json", 
            access_profile_file='config_da/annealer_v3.prf')

            solution_list = solver.minimize(**args_minimize)
            solution = solution_list.get_minimum_energy_solution()
            solution_list.print_progress()
            self.time = solution_list.solver_times.duration_solve.total_seconds()
            self.all_times = {'execution':solution_list.solver_times.duration_execution.total_seconds(),
                                'solve':solution_list.solver_times.duration_solve.total_seconds(),
                                'send_request':solution_list.solver_times.duration_send_request.total_seconds(),
                                'receive_response':solution_list.solver_times.duration_receive_response.total_seconds(),
                                'tuning':solution_list.solver_times.duration_tuning.total_seconds(),
                                'elapsed':solution_list.solver_times.duration_elapsed.total_seconds()}
            print(solution_list.solver_times)
            
        
        elif(self.solver_type=='GateQPU'):
            if not 'backend' in kwargs:
                raise ValueError('backend must be specified')
            #initialize parameters
            seed = 10
            random.seed(seed)
            theta =  [1 +  0.1 * random.uniform(-1, 1) for i in range(self.qaoa.ntheta)]

            
            np.random.seed((os.getpid() * int(time.time())) % 123456789)
            bounds = [(-2 * math.pi, 2 * math.pi) for i in range(len(theta))]

            #initialize time
            tic = time.time()
            
            total_workers = int(os.environ["OMP_NUM_THREADS"]) if "OMP_NUM_THREADS" in os.environ else 1
            # os.environ["OMP_NUM_THREADS"] = "1"


            if total_workers == 1:
                
                result = differential_evolution(
                    func=self.qaoa.get_expectation, 
                    x0 = theta, 
                    args=(kwargs['backend'], kwargs['n_shots']),
                    strategy = 'best1exp',
                    tol = kwargs['tol'],
                    callback = self.callback, 
                    popsize=1, maxiter = kwargs['maxiter'], disp=True,
                    polish= False, bounds=bounds, updating='deferred')
            
            else:

                pool=Pool(kwargs['n_workers'])                
                result = differential_evolution(
                    func=self.qaoa.get_expectation, 
                    x0 = theta, 
                    args=(kwargs['backend'], kwargs['n_shots']),
                    strategy = 'best1exp',
                    tol = kwargs['tol'],
                    callback = self.callback, 
                    popsize=1, maxiter = kwargs['maxiter'], disp=True, 
                    polish= False, bounds=bounds, updating='deferred',workers=pool.map)


            # avoid error: Object of type ndarray is not JSON serializable
            result.x = list(result.x)
            result.population = list([list(i) for i in result.population])
            result.population_energies = list(result.population_energies)
            
            #save time
            self.time = time.time()-tic
            
   
            qc_res = self.qaoa.pqc.assign_parameters(result.x)
            counts = kwargs['backend'].run (qc_res, seed_simulator = 10, shots = kwargs['n_shots']).result().get_counts()

            #solution is the most repeated outcome
            solution = [int(s) for s in max(counts,key=counts.get)][::-1]

            
            return result, solution, counts, self.best_individual


        if get_solution_list and self.solver_type!='GateQPU':
            return solution, solution_list

        return solution


