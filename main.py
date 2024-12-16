from modules.load_molecules import LoadMolecule
from modules.molecular_graph import MolecularGraph
from modules.conflict_graph import ConflictGraph
from solve_qubo import SolveQUBO
from modules.reporting import Report
from modules.post_processing import local_search, flip_and_search
import json
import time

import networkx as nx

from qiskit_aer import AerSimulator

def run(mol_name, factor, dist_threshold,removeHs=False,post_process=False,second_neighbour=False,
        collapse_rings=False,gates=False):
    # loader object
    loader = LoadMolecule('data/terpenoids')

    # load menthol & menthol's mol graph
    menthol = loader.load('menthol_bs_biblio_ob',removeHs=removeHs)
    menthol_graph = MolecularGraph(menthol,mol_name='menthol_bs_biblio_ob',collapse_rings=collapse_rings)

    results = {}
   
    times_dict = {}

    mol = loader.load(mol_name,removeHs=removeHs)
    # create molecular graph
    mol_graph = MolecularGraph(mol,mol_name=mol_name,collapse_rings=collapse_rings)

    times_dict['mol_graph'] = mol_graph.time
    # build conflict graph
    conflict_g = ConflictGraph( menthol_graph, mol_graph, second_neighbour=second_neighbour,dist_threshold=dist_threshold)

    if gates:
        nodes = [pair for pair in conflict_g.graph.nodes()]  # store original nodes to paste isolated node later on

        # check whether there are isolated nodes
        nodes_isolated = list(nx.isolates(conflict_g.graph))
        weight_isolated = []

        for i in nodes_isolated:
            weight_isolated.append(conflict_g.graph.nodes()[i]['weight'])
            conflict_g.graph.remove_node(i)
            conflict_g.var_list.remove(i)

        times_dict['conflic_graph'] = conflict_g.time

        print (mol_name, conflict_g.graph.number_of_nodes())
        if conflict_g.graph.number_of_nodes() > 1:
            
            #initializing
            n_layers = 2
            
            # solve
            solver = SolveQUBO(conflict_g=conflict_g,solver_type='GateQPU',factor=float(factor),
                                n_layers=n_layers)
            n_qubits = solver.qaoa.n_qubits
            print (n_qubits)

            #not noisy
            backend = AerSimulator()
            n_shots = 10000
            backend.shots = n_shots
            
            tol_DE = 0.0001
            maxiter_DE = 2#000 * n_qubits#5000
                    
            res_DE, solution, counts, best_individual = solver.solve_qubo(backend=backend, n_shots = n_shots, tol = tol_DE, maxiter = maxiter_DE)
            
                       
            toc = time.time()
                        
            if nodes_isolated != []:
                for i in nodes_isolated:
                    solution.insert (nodes.index(i), 1)
                    conflict_g = ConflictGraph( menthol_graph, mol_graph, second_neighbour=second_neighbour,dist_threshold=dist_threshold)

            times_dict['build_qubo'] = solver.build_qubo_time
            times_dict['solve'] = solver.time
          
            analysis = Report.from_conflict_graph(conflict_g,solution)

            conflicts = analysis.edges_in_solution

            results= {'solution': solution,
                                'similarity_size':analysis.similarity_size(delta=0.5),
                                'similarity_features':analysis.similarity_features(delta=0.5),
                                'times':times_dict,
                                'conflicts':conflicts,
                                'factor': factor}
       
            results ['counts'] = counts
            results ['n_layers'] = n_layers

        else:
            solution = [1]
            results = {'solution': solution,
                                'similarity_size':0,
                                'similarity_features':0,
                                'times':times_dict,
                                'conflicts':[]}   
               
    else:
        # solve

        solver = SolveQUBO(conflict_g=conflict_g,solver_type='QUBOSolverCPU',factor=float(factor))
        backend = None                
        solution = solver.solve_qubo(backend=backend)

        times_dict['build_qubo'] = solver.build_qubo_time
        times_dict['solve'] = solver.time
        # analysis

        solution = solution.configuration
        analysis = Report.from_conflict_graph(conflict_g,solution)

        conflicts = analysis.edges_in_solution

        results[mol_name] = {'solution': solution,
                            'similarity_size':analysis.similarity_size(delta=0.5),
                            'similarity_features':analysis.similarity_features(delta=0.5),
                            'times':times_dict,
                            'conflicts':conflicts}            

    if post_process:
        post_tic = time.time()
        new_solution = flip_and_search(conflict_graph=conflict_g,
                                    initial_sol=solution,n_times=20)
        results[mol_name]['new_solution'] = new_solution
        results[mol_name]['times']['post_process'] = time.time()-post_tic



    return results




if __name__ == '__main__':
    # dir = 'Tests/penalty_analysis'
    # for factor in np.linspace(1,1.1,11)[1:]:
    #     print(factor)
    #     results = run(factor=factor)
    #     filename = f'{dir}/factor_{factor}.json'
    #     with open(filename,'w') as f:
    #         json.dump(results,f)

    # dir = 'Tests/distance_threshold_analysis'
    # factor = 1.01
    # for dist_threshold in np.linspace(0.01,0.1,11):
    #     print(dist_threshold)
    #     results = run(factor=factor,dist_threshold=dist_threshold)
    #     filename = f'{dir}/dist_threshold_{dist_threshold}.json'
    #     with open(filename,'w') as f:
    #         json.dump(results,f)

    ############################
    # dir = 'Tests/test_hydrogens'
    # factor = 1.05
    # dist_threshold = 0.1 # relative distance
    # tic = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,add_Hs=False)
    # tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_without_Hs_both_true_normalized.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,add_Hs=True)
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {time.time()-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_with_Hs_both_true_normalized.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)

    ###############################
    # dir = 'Tests/flip_and_search'
    # factor = 1.01
    # dist_threshold = 0.1 # relative distance
    # tic = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=True,
    #               collapse_rings=False)
    # tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_no_collapse_rings.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True)
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {time.time()-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_with_Hs_both_true_normalized.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)

    ##############################
    # dir = 'Tests/second_neighbour'
    # factor = 1.02
    # dist_threshold = 0.1 # relative distance
    # tic = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=True)
    # tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_pharma_binary_second_neigh.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=False)
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {time.time()-tac} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_no_second_neigh.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)

    ##############################
    # dir = 'Tests/bigger_weight'
    # factor = 1.02
    # dist_threshold = 0.05 # relative distance
    # tic = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=True)
    # tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_without_Hs_both_true_normalized.json'
    # with open(filename,'w') as f:
    #     json.dump({'results':results,'time':tac-tic},f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=True)
    # tuc = time.time()
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {tuc-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_with_Hs_both_true_normalized.json'
    # with open(filename,'w') as f:
    #     json.dump({'results':results,'time':tuc-toc},f)

    ##############################
    dir = 'Tests/ring_simplification'
    factor = 1.01
    dist_threshold = 0.1 # relative distance
    tic = time.time()
    results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=False,
                  collapse_rings=False)
    tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_collapse_rings.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=False,
    #               collapse_rings=False)
    print(f'Total time no Hs: {tac-tic} s\n')
    # print(f' Total time with Hs {time.time()-toc} s')
    # # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_collapse_rings_pharma_binary_no_second_neigh.json'
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_collapse_rings_new_ring_weight.json'
    # filename = f'{dir}/test1.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=True,
    #               collapse_rings=True)
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {time.time()-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_no_collapse_rings_pharma_binary_second_neigh.json'
    # filename = f'{dir}/second_collapse.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)

########################################
    # dir = 'Tests/icilin'
    # factor = 1.02
    # dist_threshold = 0.1 # relative distance
    # tic = time.time()
    # results = run_icilin(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=False,
    #               collapse_rings=False)
    # tac = time.time()
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_no_collapse_rings_pharma_binary.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=True,post_process=True,second_neighbour=False,
    #               collapse_rings=True)
    # print(f'Total time no Hs: {tac-tic} s\n Total time with Hs {time.time()-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}_no_collapse_rings.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)
        
##############################
    # dir = 'Tests/bitterness2'
    # factor = 1.01
    # dist_threshold = 0.1 # relative distance
    # toc = time.time()
    # results = run_bitterness(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=False,
    #               collapse_rings=True)
    # print(f'\n Total time {time.time()-toc} s')
    # filename = f'{dir}/check_factor_{factor}_dist_threshold_{dist_threshold}_no_collapse_rings_pharma_binary_no_second_neigh.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)

##############################
    # dir = 'Tests/qaoa'
    # factor = 1.01
    # dist_threshold = 0.1 # relative distance
    # toc = time.time()
    # results = run(factor=factor,dist_threshold=dist_threshold,removeHs=False,post_process=True,second_neighbour=False,
    #               collapse_rings=True,gates=True)
    # print(f'\n Total time {time.time()-toc} s')
    # filename = f'{dir}/factor_{factor}_dist_threshold_{dist_threshold}.json'
    # with open(filename,'w') as f:
    #     json.dump(results,f)