from modules.conflict_graph import ConflictGraph
from modules.utils import validate_constraints,get_matching_nodes, similarity_features,similarity_size

def clean_solution(solution,edges_in_solution,var_list,graph):
    '''
    Clean a solution by removing the node with the smallest weight for each conflict.
    Attributes:
        - solution: solution to clean
        - edges_in_solution: list of edges in the solution that are in conflict
        - var_list: list of variables in the solution
        - graph: conflict graph
    '''
    # solution = solution.copy()
    removed_nodes = []
    while len(edges_in_solution) >0:
        edge = edges_in_solution[0]
        if graph.nodes(data=True)[edge[0]]['weight'] > graph.nodes(data=True)[edge[1]]['weight']:
            index = var_list.index(edge[1])
            solution[index] = 0
            removed_nodes.append(edge[1])
        else:
            index = var_list.index(edge[0])
            solution[index] = 0
            removed_nodes.append(edge[0])
        matching_nodes = get_matching_nodes(solution=solution,var_list=var_list)
        edges_in_solution = validate_constraints(matching_nodes=matching_nodes,graph=graph)
    
    return solution,removed_nodes


def activate_node_in_solution(node,solution,var_list):
    '''
    Activate a node in a solution.
    Attributes:
        - node: node to activate
        - solution: solution to activate node in
        - var_list: list of variables in the solution
    '''
    node_index = var_list.index(node)
    solution[node_index] = 1
    return solution


def get_weight(conflict_graph:ConflictGraph,solution):
    '''
    Compute the weight of a solution.
    Attributes:
        - conflict_graph
        - solution'''
    weight = 0
    # print(conflict_graph.graph.nodes())
    # print(conflict_graph.var_list)
    for i,s in enumerate(solution):
        if s:
            weight += conflict_graph.graph.nodes(data=True)[conflict_graph.var_list[i]]['weight']
    return weight

  
def local_search(conflict_graph:ConflictGraph,initial_sol,search_order=[]):
    '''
    Attributes:
        - conflict_graph
        - initial_sol
        - search_order: it MUST be a permutation of the indices from 0 to n_nodes-1
    '''
    if search_order == []:
        search_order = [i for i in range(conflict_graph.graph.number_of_nodes())]
    

    #fetch data
    matching_nodes = get_matching_nodes(solution=initial_sol,var_list=conflict_graph.var_list)
    edges_in_solution = validate_constraints(matching_nodes=matching_nodes,graph=conflict_graph.graph)

    removed_nodes = []
    if len(edges_in_solution) > 0:
        initial_sol,removed_nodes = clean_solution(solution=initial_sol,edges_in_solution=edges_in_solution,
                                                    var_list=conflict_graph.var_list,graph=conflict_graph.graph)
        matching_nodes = list(set(matching_nodes).difference(set(removed_nodes)))    
    
    #intial sol objective value
    initial_objective_value = get_weight(conflict_graph=conflict_graph,solution=initial_sol)

    # loop over nodes to try to add them to the solution
    for node_pair_index in search_order:
        node_pair = conflict_graph.var_list[node_pair_index]
        if node_pair in matching_nodes or node_pair in removed_nodes:
            continue
        conflict = False
        for other_node_pair in matching_nodes:
            if conflict_graph.graph.has_edge(node_pair,other_node_pair):
                conflict = True
                continue

        if not conflict:
            # print(node_pair,initial_sol)
            new_solution = activate_node_in_solution(node=node_pair,solution=initial_sol,
                                                                   var_list=conflict_graph.var_list)
            # calculate new weight 
            new_objective_value = get_weight(conflict_graph=conflict_graph,solution=new_solution)
            # check if new similarity improves old one and keep solution if so
            if new_objective_value > initial_objective_value: # we do it over the conflict graph, weights are +sim, so we want to maximize
                initial_sol = new_solution.copy()
                initial_objective_value = new_objective_value
                matching_nodes = get_matching_nodes(solution=initial_sol,var_list=conflict_graph.var_list)
    
    return initial_sol


def flip_and_search(conflict_graph:ConflictGraph,initial_sol,n_times=5,search_order=[]):
    '''
    Flip n_times nodes and search locally for a better solution.
    Attributes:
        - conflict_graph
        - initial_sol
        - n_times: number of nodes to flip and search
        - search_order: it MUST be a permutation of the indices from 0 to n_matching_nodes-1
    '''
    # first local search
    initial_sol = local_search(conflict_graph=conflict_graph,initial_sol=initial_sol,search_order=search_order)
    matching_nodes = get_matching_nodes(solution=initial_sol,var_list=conflict_graph.var_list)
    edges_in_solution = validate_constraints(matching_nodes=matching_nodes,graph=conflict_graph.graph)

    removed_nodes = []
    if len(edges_in_solution) > 0:
        initial_sol,removed_nodes = clean_solution(solution=initial_sol,edges_in_solution=edges_in_solution,
                                                    var_list=conflict_graph.var_list,graph=conflict_graph.graph)
        matching_nodes = list(set(matching_nodes).difference(set(removed_nodes)))   

    #intial sol objective value
    initial_objective_value = get_weight(conflict_graph=conflict_graph,solution=initial_sol)

    best_sol = initial_sol.copy()
    best_objective_value = initial_objective_value
    n_times = min(n_times,len(matching_nodes))
    for iter in range(n_times):
        new_solution = initial_sol.copy()
        new_solution[conflict_graph.var_list.index(matching_nodes[iter])] = 0
        # print(matching_nodes[iter])
        # print(get_matching_nodes(solution=new_solution,var_list=conflict_graph.var_list))
        new_solution = local_search(conflict_graph=conflict_graph,initial_sol=new_solution,
                                    search_order=search_order)
        # print(get_matching_nodes(solution=new_solution,var_list=conflict_graph.var_list))
        new_objective_value = get_weight(conflict_graph=conflict_graph,solution=new_solution)

        if new_objective_value > best_objective_value:
            # print(iter)
            best_sol = new_solution.copy()
            best_objective_value = new_objective_value
    
    return best_sol