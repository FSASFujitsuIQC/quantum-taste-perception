import numpy as np
import itertools
import json
import csv
from modules.node import Ring


def get_rings_in_solution(matching_nodes):
    ''' Get the rings in a solution.
    Attributes:
        - matching_nodes: list of matching nodes\\
        Returns:
        - rings_in_solution_mol1: list of rings in molecule 1
        - rings_in_solution_mol2: list of rings in molecule 2
    '''
    rings_in_solution_mol1 = []
    rings_in_solution_mol2 = []
    # rings_in_solution = []
    for node_pair in matching_nodes:
        if type(node_pair[0]) == str:  # atom nodes have int label, rings have str label ('r0')
            rings_in_solution_mol1.append(node_pair[0])
            rings_in_solution_mol2.append(node_pair[1])
            # rings_in_solution.append(node_pair)
    return rings_in_solution_mol1, rings_in_solution_mol2

# def get_overlapping_rings(matching_nodes):
#     rings_in_solution_mol1, rings_in_solution_mol2 = get_rings_in_solution(matching_nodes)
#     for ring in rings_in_solution_mol1:
#         continue

def get_number_of_atoms_in_rings(matching_nodes,mol1_graph,mol2_graph):
    ''' Get the number of non repeating atoms in the rings in a solution.
    
    Args:
        - matching_nodes: list of matching nodes
        - mol1_graph: graph of molecule 1
        - mol2_graph: graph of molecule 2
    
    Returns:
        - nb_atoms_mol1: number of atoms in the rings in molecule 1
        - nb_atoms_mol2: number of atoms in the rings in molecule 2
    '''
    rings_in_solution_mol1, rings_in_solution_mol2 = get_rings_in_solution(matching_nodes)
    atoms_mol1 = set()
    for ring in rings_in_solution_mol1: 
        atoms_mol1.update(mol1_graph.rings[ring])
    nb_atoms_mol1 = len(atoms_mol1)

    atoms_mol2 = set()
    for ring in rings_in_solution_mol2: 
        atoms_mol2.update(mol2_graph.rings[ring])
    nb_atoms_mol2 = len(atoms_mol2)

    return nb_atoms_mol1,nb_atoms_mol2



############################################
## DECODING SOLUTION & GETTING SIMILIRATY ##
############################################
def get_matching_nodes(solution,var_list):
        ''' Get the matching nodes in a solution.

        Args:
            - solution: solution to decode
            - var_list: list of variables in the solution

        Returns:
            - matching_nodes: list of matching nodes
        '''
        matching_nodes = []
        for i,s in enumerate(solution):
            if s:
                matching_nodes.append(var_list[i])
        
        return matching_nodes


def validate_constraints(matching_nodes,graph):
    ''' Validate the constraints of a solution.

    Args:
        - matching_nodes: list of matching nodes
        - graph: conflict graph
    
    Returns:
        - edges_in_solution: list of edges in the solution
    '''
    edges_in_solution = []
    # matching_nodes = matching_nodes(solution,var_list)
    for edge in itertools.combinations(matching_nodes,r=2):
        # check if edge is conflict edge (=is part of conflict graph)
        if graph.has_edge(*edge):
            edges_in_solution.append(edge)

    return edges_in_solution


def get_matching_features(matching_nodes,mol1_graph,mol2_graph):
    ''' Get the matching features in a solution (the joint weight of the matching
    nodes).

    Args:
        - matching_nodes: list of matching nodes
        - mol1_graph: graph of molecule 1
        - mol2_graph: graph of molecule 2

    Returns:
        - matching_features: joint weight of the matching nodes
    '''
    #initialize
    matching_features_atoms = 0
    matching_features_rings = 0
    total_ring_weight = 0
    for node_pair in matching_nodes:
        #fetch data
        node1 = node_pair[0]
        node2 = node_pair[1]
        node1_ft = mol1_graph.mol_graph.nodes[node1]['features']
        node2_ft = mol2_graph.mol_graph.nodes[node2]['features']

        #calculate weight
        if type(node1_ft) == Ring:
            #rings have a weight proportional to the number of atoms in the ring
            weight = min(sum(list(node1_ft.atomic_nb.values())),sum(list(node2_ft.atomic_nb.values())))
            # weight = sum(list(node1_ft.atomic_nb.values()))
            total_ring_weight += weight
            matching_features_rings += weight*node1_ft.compare(node2_ft)
        else:
            #atoms have a weight of 1
            weight = 1
            matching_features_atoms += weight*node1_ft.compare(node2_ft)

    #take care of normalization
    if total_ring_weight == 0:
        rescale = 1
    else:
        nb_atoms_in_rings = min(get_number_of_atoms_in_rings(matching_nodes,mol1_graph,mol2_graph))
        # nb_atoms_in_rings = get_number_of_atoms_in_rings(matching_nodes,mol1_graph,mol2_graph)[0]
        rescale = nb_atoms_in_rings/total_ring_weight

    return matching_features_atoms +  rescale * matching_features_rings


def similarity_size(solution,mol1_graph,mol2_graph,delta):
    ''' Compute the metric similarity size of two molecules.

    Args:
        - solution: solution to decode
        - mol1_graph: graph of molecule 1
        - mol2_graph: graph of molecule 2
        - delta: relative importance of the size of the molecules

    Returns:
        - similarity: metric similarity size
    '''
    if delta < 0 or delta > 1:
        raise ValueError('delta must be in [0,1].')
    
    # fetch data
    matching_nodes = sum(solution)
    n_nodes1 = mol1_graph.mol_graph.number_of_nodes()
    n_nodes2 = mol2_graph.mol_graph.number_of_nodes()

    # ratios
    alpha = min(matching_nodes/n_nodes1,matching_nodes/n_nodes2)
    beta = max(matching_nodes/n_nodes1,matching_nodes/n_nodes2)
    
    return delta*alpha + (1-delta)*beta

def similarity_features(solution,mol1_graph,mol2_graph,var_list,delta):
    ''' Compute the metric similarity features of two molecules.

    Args:
        - solution: solution to decode
        - mol1_graph: graph of molecule 1
        - mol2_graph: graph of molecule 2
        - var_list: list of tuple of nodes. Each tuple is mapped to the variable given by its index in the list
        - delta: relative importance of the size of the molecules
    
    Returns:
        - similarity: metric similarity features
    '''
    if delta < 0 or delta > 1:
        raise ValueError('delta must be in [0,1].')
    
    #fetch data
    total_features_mol1 = mol1_graph.get_total_features()
    total_features_mol2 = mol2_graph.get_total_features()
    matching_nodes = get_matching_nodes(solution=solution,var_list=var_list)
    matching_features = get_matching_features(matching_nodes=matching_nodes,
                                              mol1_graph=mol1_graph,mol2_graph=mol2_graph)

    #ratios
    alpha = min(matching_features/total_features_mol1,matching_features/total_features_mol2)
    beta = max(matching_features/total_features_mol1,matching_features/total_features_mol2)
    
    return delta*alpha + (1-delta)*beta


############################################
############# QUALITY METRICS ##############
# ############################################
# def cluster1d(x):
#     '''
#     1-D clustering following answers to this question
#       https://stackoverflow.com/questions/11513484/1d-number-array-clustering
#       Args:
#         - x: sorted np.array in DESCENDING ORDER
#     '''
#     diff = [x[i] - x[i-1] for i in range(1, len(x))]
#     rel_diff = [diff[i]/x[i] for i in range(len(diff))]
#     arg = argrelextrema(np.array(rel_diff), np.less)[0]
#     initial = 0
#     indices = []
#     for extr in arg:
#         indices.append([initial,extr+1])
#         initial = extr + 1
#     indices.append([initial,len(x)])
#     return indices

# def kcluster1d(x,k):
#     '''
#     1-D clustering
#         Args:
#             - x: sorted np.array in DESCENDING ORDER
#             - k: number of clusters
#         '''
#     jnb = JenksNaturalBreaks(k)
#     jnb.fit(x)
#     return jnb.labels_, jnb.groups_

def group_metric(exp_groups,sim_group):
    n_groups = len(exp_groups)
    error = {}
    for i_group in range(n_groups):
        for mol_name in exp_groups[i_group]:
            if mol_name in sim_group[i_group]:
                error[mol_name] = 0
            else:
                for i in range(n_groups-1):
                    if mol_name in sim_group[(i_group+1+i)%n_groups]:
                        error[mol_name] = 1+i
    return error

groups_lab=[['142218_icilin',  'CID_2758_eucalyptol',  'CID_6616_camphene',  'CID_10106_isocineole',  'CID_443162_l-alpha-terpineol',  'CID_20055523_trans-4-thujanol',  'CID_111037_alpha-terpinylacetate',  'CID_11230_terpinen-4-ol',  'CID_2537_2-bornanone',  'CID_82227_alphapinene',  'CID_440967_betapinene',  'CID_28930_alphafenchene',  'CID_527424_p-Mentha-1,5,8-triene',  'CID_91710638_beta-Z-curcumen-12-ol',  'CID_101412242_beta-Copaene-4-alpha-ol'], ['CID_1549025_cis-geranylacetate',  'CID_567757_verbenyl-ethyl-ether',  'CID_91746870_selina-1,4-diene',  'CID_10812_betacymene',  'CID_10703_o-cymene',  'CID_6432312_gamma-elemene',  'CID_94254_elixene',  'CID_10657_beta-cadinene',  'CID_11463_alpha-terpinolene',  'CID_5315347_beta-cyclogermacrane',  'CID_7462_alpha-terpinene',  'CID_261491_alpha-thujone',  'CID_6918391_beta-elemene',  'CID_442348_alpha-cedrene',  'CID_92139_alpha-curcumene',  'CID_7461_gamma-terpinene',  'CID_28237_beta-selinene',  'CID_18818_sabinene',  'CID_17868_alphathujene',  'CID_11127403_alpha-zingiberene',  'CID_11106487_beta-sesquiphellandrene',  'CID_11106485_beta-cedrene',  'CID_442359_alpha-cubebene',  'CID_522296_Selina-3,7-11-diene',  'CID_518814_beta-cuvebene',  'CID_11401461_cis-tujopsene'], ['CID_65575_cedrol',  'CID_6549_linalool',  'CID_160799_tau-cadinol',  'CID_335_o-cresol',  'CID_240122_guaiyl-acetate',  'CID_30247_lavandulyl-acetate',  'CID_5365847_ethyl-geranyl-ether',  'CID_440917_d-limonene',  'CID_441005_delta-cadinene',  'CID_12304570_silvestrene',  'CID_68140_psilimonene',  'CID_6432308_gamma-muurolene',  'CID_53359349_sesquithujene',  'CID_102443_isoterpinolene',  'CID_5368451_cosmene',  'CID_68316_perillene',  'CID_530421_aristolene',  'CID_31289_nonanal',  'CID_28481_calarene',  'CID_12343_1,4-cyclohexadiene',  'CID_85582292_24thujadiene'], ['CID_91723677_cadina-3,5-diene',  'CID_15094_gamma-cadinene',  'CID_6429022_trans-calamenene',  'CID_7855_2-propenenitrile',  'CID_5281520_humulene',  'CID_12306048_alpha-cadinene',  'CID_26049_3-carene',  'CID_637563_anethole',  'CID_5281515_caryophyllene',  'CID_12222_2-propynal',  '4444855_isocaryophyllene',  '4444848_beta-caryophyllene',  'CID_12302243_alpha-calarorene',  'CID_5371125_Neo-allo-ocimene']]
groups_lab_menthol=[['menthol_bs_biblio_ob','142218_icilin',  'CID_2758_eucalyptol',  'CID_6616_camphene',  'CID_10106_isocineole',  'CID_443162_l-alpha-terpineol',  'CID_20055523_trans-4-thujanol',  'CID_111037_alpha-terpinylacetate',  'CID_11230_terpinen-4-ol',  'CID_2537_2-bornanone',  'CID_82227_alphapinene',  'CID_440967_betapinene',  'CID_28930_alphafenchene',  'CID_527424_p-Mentha-1,5,8-triene',  'CID_91710638_beta-Z-curcumen-12-ol',  'CID_101412242_beta-Copaene-4-alpha-ol'], ['CID_1549025_cis-geranylacetate',  'CID_567757_verbenyl-ethyl-ether',  'CID_91746870_selina-1,4-diene',  'CID_10812_betacymene',  'CID_10703_o-cymene',  'CID_6432312_gamma-elemene',  'CID_94254_elixene',  'CID_10657_beta-cadinene',  'CID_11463_alpha-terpinolene',  'CID_5315347_beta-cyclogermacrane',  'CID_7462_alpha-terpinene',  'CID_261491_alpha-thujone',  'CID_6918391_beta-elemene',  'CID_442348_alpha-cedrene',  'CID_92139_alpha-curcumene',  'CID_7461_gamma-terpinene',  'CID_28237_beta-selinene',  'CID_18818_sabinene',  'CID_17868_alphathujene',  'CID_11127403_alpha-zingiberene',  'CID_11106487_beta-sesquiphellandrene',  'CID_11106485_beta-cedrene',  'CID_442359_alpha-cubebene',  'CID_522296_Selina-3,7-11-diene',  'CID_518814_beta-cuvebene',  'CID_11401461_cis-tujopsene'], ['CID_65575_cedrol',  'CID_6549_linalool',  'CID_160799_tau-cadinol',  'CID_335_o-cresol',  'CID_240122_guaiyl-acetate',  'CID_30247_lavandulyl-acetate',  'CID_5365847_ethyl-geranyl-ether',  'CID_440917_d-limonene',  'CID_441005_delta-cadinene',  'CID_12304570_silvestrene',  'CID_68140_psilimonene',  'CID_6432308_gamma-muurolene',  'CID_53359349_sesquithujene',  'CID_102443_isoterpinolene',  'CID_5368451_cosmene',  'CID_68316_perillene',  'CID_530421_aristolene',  'CID_31289_nonanal',  'CID_28481_calarene',  'CID_12343_1,4-cyclohexadiene',  'CID_85582292_24thujadiene'], ['CID_91723677_cadina-3,5-diene',  'CID_15094_gamma-cadinene',  'CID_6429022_trans-calamenene',  'CID_7855_2-propenenitrile',  'CID_5281520_humulene',  'CID_12306048_alpha-cadinene',  'CID_26049_3-carene',  'CID_637563_anethole',  'CID_5281515_caryophyllene',  'CID_12222_2-propynal',  '4444855_isocaryophyllene',  '4444848_beta-caryophyllene',  'CID_12302243_alpha-calarorene',  'CID_5371125_Neo-allo-ocimene']]

def get_mol_per_group_in_top_k(ranking,exp_groups=groups_lab,k=14):
    '''
    Get the number of molecules per group in the top k molecules.
    Attributes:
        - ranking: ranking of molecules
        - exp_groups: list of groups of molecules
        - k: number of molecules to consider
    '''
    unified_groups_lab=list(itertools.chain.from_iterable(exp_groups))   
    ranking = {mol:ranking[mol] for mol in ranking.keys() if mol in unified_groups_lab}
    top_k = list(ranking.keys())[:k]
    top_k_groups = {}
    for i,group in enumerate(exp_groups):
        top_k_groups[i+1] = []
        for mol in top_k:
            if mol in group:
                top_k_groups[i+1].append(mol)
    return top_k_groups

# def group_pos_metric(exp_groups,sim_group):
#     n_groups = len(exp_groups)
#     joint = []
#     for group in exp_groups: joint += group

#     error = {}
#     for i_group in range(n_groups):
#         for mol_name in exp_groups[i_group]:
#             if mol_name in sim_group[i_group]:
#                 error[mol_name] = 0
#             else:
#                 for i in range(n_groups-1):
#                     if mol_name in sim_group[(i_group+1+i)%n_groups]:
#                         if (i_group+1+i)%n_groups < i_group:
#                             error[mol_name] = 

############################################
############# RESULTS TO CSV ###############
############################################
def json_to_ranking(json_path, post_process=False, DA=False):
    '''
    Read json file and return ranking of molecules.
    Attributes:
        - json_path: path to json file
        - post_process: whether to consider post processed results
        - DA: whether results come from DA via run_parallel.py script\\
        Returns:
        - ranking_per_sim_size: ranking of molecules by similarity size
        - ranking_per_sim_feat: ranking of molecules by similarity features
    '''
    #read json
    with open(json_path) as json_file:
        results = json.load(json_file)
    if DA: results = results['results']
    #get ranking
    ranking_per_sim_size = {}
    ranking_per_sim_feat = {}
    for mol_name,value in results.items():
        ranking_per_sim_size[mol_name] = value['similarity_size'+('_post' if post_process else '')]
        ranking_per_sim_feat[mol_name] = value['similarity_features'+('_post' if post_process else '')]
    #sort ranking
    ranking_per_sim_size = dict(sorted(ranking_per_sim_size.items(),key=lambda x:x[1],reverse=True))
    ranking_per_sim_feat = dict(sorted(ranking_per_sim_feat.items(),key=lambda x:x[1],reverse=True))
    return ranking_per_sim_size, ranking_per_sim_feat

def ranking_to_csv(ranking:dict,results_path):
    '''
    Write results to csv file.
    Attributes:
        - results: dictionary of results
        - results_path: path to save results to
    '''
    if not results_path.endswith('.csv'):
        results_path += '.csv'
    
    #write ranking to csv
    with open(results_path, 'w') as f:
        w = csv.writer(f,delimiter=';',lineterminator='\n')
        w.writerow(['Molecule','Similarity to menthol'])
        for mol_name,sim in ranking.items():
            w.writerow([mol_name,sim])