import itertools
import time
from typing import Union
import networkx as nx
from modules.molecular_graph import MolecularGraph
from modules.node import Atom, Ring
from modules.config import CRITICAL_FEATURES, DISTANCE_THRESHOLD
import numpy as np

class ConflictGraph():
    '''
        Class to build a conflict graph for structure-based molecular
        comparison. Args:
        - mol1_graph: MolecularGraph object of molecule 1
        - mol2_graph: MolecularGraph object of molecule 2
        - graph: nx.Graph object of the conflict graph
        - second_neighbour: bool, whether to check for second neighbour conflicts
        - dist_threshold: float, threshold for distance conflict
    '''
    def __init__(self,mol1_graph:MolecularGraph
                 ,mol2_graph:MolecularGraph,
                 graph = False,
                 second_neighbour=False,
                 dist_threshold = DISTANCE_THRESHOLD,
                 var_list=False) -> None:
        # initialize
        self.mol1_graph = mol1_graph
        self.mol2_graph = mol2_graph
        self.dist_threshold = dist_threshold
        build_graph_time = 0
        # build graph if not provided (bc used in reporting too)
        if not graph:
            start_time = time.time()
            graph = self._build_conflict_graph(second_neighbour=second_neighbour)
            build_graph_time = time.time()-start_time
        self.graph = graph
        if not var_list:
            var_list = self._get_var_list()
        self.var_list = var_list
        self.time = build_graph_time

    def _compatible(self,node1:Union[Atom,Ring],node2:Union[Atom,Ring]):
        ''' For the pharmacophoric features, we require that one
        of the features is True for both nodes, not that they are equal.
        
        Args:
            - node1: Atom or Ring object
            - node2: Atom or Ring object

        Returns:
            - bool, whether the nodes are compatible or not'''

        if type(node1) != type(node2): # only rings with rings 
            # print('different type')
            return False
        
        if type(node1) == Ring:
            # print('aromatic vs non aromatic ring')
            if node1.is_aromatic != node2.is_aromatic: return False  # only aromatic rings to aromatic rings
            # if sum(list(node1.atomic_nb.values())) != sum(list(node2.atomic_nb.values())): return False
            # if node1.double != node2.double: return False

        pharmacophoric = ['is_donor','is_acceptor',
                     'is_hydrophobic','is_aromatic']
        for feature in CRITICAL_FEATURES:
            if node1.__getattribute__(feature) == []:
                continue
            if feature in pharmacophoric:
                if node1.__getattribute__(feature) and node2.__getattribute__(feature):
                    return True
            else:
                if node1.__getattribute__(feature) == node2.__getattribute__(feature):
                    return True
        ###### REFINAR ESTO: si elementos en bond order están diferente orden ya 
        # sería falso. Además, esta condición solo tendría sentido cuando ponemos 
        # como propiedad crítica una de la farmacofóricas
        if node1.__dict__ == node2.__dict__: 
            return True
        else:
            # print('no match in critical features')
            return False
    
    def _check_conflict(self,node_pair,second_neighbour=False):
        ''' Check if there is a conflict between two nodes. Args:
            - node_pair: tuple of two nodes
            - second_neighbour: bool, whether to check for second neighbour conflicts\\
        Returns:
            - conflict: bool, whether there is a conflict or not
            - conflict_type: list of conflict types
        '''
        mol1_graph = self.mol1_graph.mol_graph
        mol2_graph = self.mol2_graph.mol_graph
        #initialize
        conflict = False
        conflict_type = []

        #fetch edge indices
        mol1_edge_idx = (node_pair[0][0],node_pair[1][0])
        mol2_edge_idx = (node_pair[0][1],node_pair[1][1])

        # check bijection contraint
        if (mol1_edge_idx[0] == mol1_edge_idx[1]) or (mol2_edge_idx[0]==mol2_edge_idx[1]):
            # print(f'Bijection conflict, edge {node_pair}')
            # conflict = True
            conflict_type.append('bijection')

        if (mol1_graph.has_edge(*mol1_edge_idx) != mol2_graph.has_edge(*mol2_edge_idx)):
            # print(f'Has edge conflict, edge {node_pair}')

            # conflict = True
            conflict_type.append('edge')

        elif mol1_graph.has_edge(*mol1_edge_idx): #both are true if we are at this point
            #compare bond types
            diff_bond_type = bool(mol1_graph.get_edge_data(*mol1_edge_idx)['bond_type'] !=
                            mol2_graph.get_edge_data(*mol2_edge_idx)['bond_type'])
            # compare bond distance
            dist_1 = mol1_graph.get_edge_data(*mol1_edge_idx)['distance']
            dist_2 = mol2_graph.get_edge_data(*mol2_edge_idx)['distance']
            relative_difference_dist = abs(dist_1-dist_2)/dist_1

            incompatible_dist = bool(relative_difference_dist>self.dist_threshold)

            if diff_bond_type:
                # print(f'Diff bond type {diff_bond_type}, incompat dist {incompatible_dist} conflict, edge {node_pair}')
                # conflict = True
                conflict_type.append('bond_type')
        # dist_1 = np.linalg.norm(self.mol1_graph.pos[mol1_edge_idx[0]]-self.mol1_graph.pos[mol1_edge_idx[1]])
        # dist_2 = np.linalg.norm(self.mol2_graph.pos[mol2_edge_idx[0]]-self.mol2_graph.pos[mol2_edge_idx[1]])
        # relative_difference_dist = abs(dist_1-dist_2)/dist_1
        # incompatible_dist = bool(relative_difference_dist>self.dist_threshold)
            if incompatible_dist:
                # conflict = True
                conflict_type.append('distance')
        
        # check 2nd neighbour conflict
        # if second_neighbour and ('bijection' not in conflict_type or 'edge' not in conflict_type):
        if second_neighbour and conflict_type==[]:
            # print(node_pair)
            second_neighbours1 = set(mol1_graph.neighbors(mol1_edge_idx[0])
                                     ).intersection(set(mol1_graph.neighbors(mol1_edge_idx[1])))
            second_neighbours2 = set(mol2_graph.neighbors(mol2_edge_idx[0])
                                     ).intersection(set(mol2_graph.neighbors(mol2_edge_idx[1])))
            # print(second_neighbours1,second_neighbours2)
            if len(second_neighbours1) != len(second_neighbours2):
                conflict_type.append('second_neighbour')
                # print(f'second neigh conflict: {node_pair}')

        if len(conflict_type) > 0:
            conflict = True

        return conflict,conflict_type

    def _build_conflict_graph(self,second_neighbour=False):
        ''' Build the conflict graph. Args:
            - second_neighbour: bool, whether to check for second neighbour conflicts\\
        Returns:
            - conflict_G: nx.Graph object of the conflict graph
        '''
        mol1_graph = self.mol1_graph.mol_graph
        mol2_graph = self.mol2_graph.mol_graph
        # instantiate graph
        conflict_G = nx.Graph()

        # add nodes according to rule
        for node_mol1 in mol1_graph.nodes:
            node_mol1_ft = mol1_graph.nodes[node_mol1]['features']

            for node_mol2 in mol2_graph.nodes:
                node_mol2_ft = mol2_graph.nodes[node_mol2]['features']
                # print(node_mol1,node_mol2)
                if self._compatible(node_mol1_ft,node_mol2_ft): # check if nodes are compatible
                    weight = node_mol1_ft.compare(node_mol2_ft)

                    # increase weight if node is a ring
                    if type(node_mol1_ft) == Ring:
                        weight *= sum(list(node_mol1_ft.atomic_nb.values()))
                        # weight *= 1.5
                        # print(sum(list(node_mol1_ft.atomic_nb.values())),'    ',weight)

                    # add node with weight
                    conflict_G.add_node((node_mol1,node_mol2),weight=weight)

        # check for conflicts and add edges if necessary
        for node_pair in itertools.combinations(conflict_G.nodes,r=2): 
            # print(node_pair) 
            conflict,conflict_type = self._check_conflict(node_pair,second_neighbour)     
            # if type(conflict) != bool:
            # print(node_pair,'   ',conflict)
            
            if conflict: # add edge if there is a conflict
                edge_weight = min(conflict_G.nodes[node_pair[0]]['weight'],
                                    conflict_G.nodes[node_pair[1]]['weight'])
                conflict_G.add_edge(*node_pair,weight=edge_weight, type=conflict_type)

        return conflict_G
    
    def _get_var_list(self):
        ''' Get the list of variables. Each tuple of nodes from molecular graphs is 
         identified by the index of the node in the conflict graph.
          Returns:
            - var_list: list of tuples of nodes'''
        return [node for node in self.graph.nodes]
        # return {i:node for i,node in enumerate(self.graph.nodes)}