from modules.molecular_graph import MolecularGraph
from modules.conflict_graph import ConflictGraph
from modules.config import WEIGHTING_SCHEME
from modules.utils import validate_constraints, get_matching_nodes, get_matching_features, similarity_size, similarity_features
from modules.post_processing import clean_solution
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

class Report(ConflictGraph):
    '''
        Class to report the solution of a conflict graph. 
        
        Args:
        - mol1_graph: MolecularGraph object of molecule 1
        - mol2_graph: MolecularGraph object of molecule 2
        - solution: list of 0s and 1s, where 1 means that the corresponding node is part of the solution
        - graph: nx.Graph object of the conflict graph
    '''
    def __init__(self, mol1_graph: MolecularGraph, mol2_graph: MolecularGraph,solution,
                 graph=False,var_list=False) -> None:
        super().__init__(mol1_graph, mol2_graph,graph=graph,var_list=var_list)
        self.solution = solution
        self.matching_nodes = self._matching_nodes()
        self.edges_in_solution = self._validate_constraints()

    @classmethod
    def from_conflict_graph(cls,conflict_graph:ConflictGraph,solution):
        '''
            Create a Report object from a ConflictGraph object. Args:
            - conflict_graph: ConflictGraph object
            - solution: list of 0s and 1s, where 1 means that the corresponding node is part of the solution
        '''
        return cls(conflict_graph.mol1_graph,conflict_graph.mol2_graph,
                   graph=conflict_graph.graph,var_list=conflict_graph.var_list,solution=solution)

    def similarity_size(self,delta):
        '''
            Similarity size metric. Args:
            - delta: float between 0 and 1. If delta=0, the metric is equal to the ratio of matching nodes in the smaller molecule.
            If delta=1, the metric is equal to the ratio of matching nodes in the larger molecule.
        '''
       
        return similarity_size(self.solution,self.mol1_graph,self.mol2_graph,delta)

    def similarity_features(self,delta):
        '''
            Similarity features metric. Args:
            - delta: float between 0 and 1. If delta=0, the metric is equal to the ratio of matching features in the smaller molecule.
            If delta=1, the metric is equal to the ratio of matching features in the larger molecule.
        '''
        
        return similarity_features(self.solution,self.mol1_graph,self.mol2_graph,self.var_list,delta)

    def _matching_features(self):
        '''
            Get the matching features in the solution. Quantity proportional to the matching nodes in the solution.
        '''
        return get_matching_features(self.matching_nodes,self.mol1_graph,self.mol2_graph)
    
    def _matching_nodes(self):
        '''
            Get the matching nodes in the solution
        '''
        return get_matching_nodes(solution=self.solution,var_list=self.var_list)
    
    def _validate_constraints(self):
        '''
            Get the edges in the solution that violate the constraints
        '''
        return validate_constraints(self.matching_nodes,self.graph)
    
    def clean_solution(self):
        '''
            For each conflict, remove the matching with the smallest weight
        '''
        return clean_solution(self.solution,self.edges_in_solution,self.var_list,self.graph)

    def plot_solution(self,labels=False,return_fig=False):
        '''
            Plot the solution. Args:
            - labels: if True, plot the node labels
        '''
        fig, ax = plt.subplots(nrows=1,ncols=2)
        options = {"edgecolors": "tab:gray", "node_size": 80, "alpha": 1}
        #set colors
        color_grad = [plt.get_cmap('nipy_spectral')(i) for i in np.linspace(0,1,len(self.matching_nodes))]

        print('COLOR SCHEME:')
        print('Colored nodes are part of the largest common substructure between molecules.')
        print('Nodes with the same color are matching nodes in the solution being plotted.')
        print('WHITE nodes are NOT part of the largest commom substructure.')
        for i,mol_graph in enumerate([self.mol1_graph,self.mol2_graph]):
            #get matching and non-matching nodes per molecule
            matching_nodes_mol_i = [node_pair[i] for node_pair in self.matching_nodes]
            other_nodes_mol_i = set(mol_graph.mol_graph.nodes).difference(set(matching_nodes_mol_i))
            # set position
            pos = nx.kamada_kawai_layout(mol_graph.mol_graph)

            #matching nodes
            nx.draw_networkx_nodes(mol_graph.mol_graph,pos=pos,nodelist=matching_nodes_mol_i,node_color=color_grad,ax=ax[i],**options)
            #non-matching nodes
            nx.draw_networkx_nodes(mol_graph.mol_graph,pos=pos,nodelist=other_nodes_mol_i,node_color='white',ax=ax[i],**options)
            #edges
            nx.draw_networkx_edges(mol_graph.mol_graph,pos=pos,ax=ax[i])
            if labels:
                pos_higher = {}
                for node,(x,y) in pos.items():
                    pos_higher[node]=(x+0.07,y+0.07)
                nx.draw_networkx_labels(mol_graph.mol_graph,pos=pos_higher,ax=ax[i])
            ax[i].axis("off")
            ax[i].set_title(mol_graph.name)

        if return_fig:
            return fig
        
        plt.tight_layout()
        # plt.axis("off")
        plt.show()

    def full_report(self,delta,plot=True,print_conflicts=True,labels=False,clean=True):
        '''
            Print a full report of the solution. Args:
            - delta: float between 0 and 1. If delta=0, the metric is equal to the ratio of matching features in the smaller molecule.
            If delta=1, the metric is equal to the ratio of matching features in the larger molecule.
            - plot: if True, plot the solution
            - print_conflicts: if True, print the conflicts in the solution
            - labels: if True, plot the node labels
            - clean: if True, postprocess the solution
        '''
        if plot:
            self.plot_solution(labels=labels)

        print(f'Similarity size: {self.similarity_size(delta):.4f}'
              f' | Similarity features: {self.similarity_features(delta):.4f}')
        print(f'There are {len(self.edges_in_solution)} conflicts in the solution')

        if print_conflicts and len(self.edges_in_solution)>0:
            for edge in self.edges_in_solution:
                print(f'Matching {edge[0]} is in conflict with matching {edge[1]}, Type: {self.graph.get_edge_data(*edge)["type"]}')
        
        # postprocess solution if there are conflicts
        if len(self.edges_in_solution) > 0 and clean:
            self.solution,removed_nodes = self.clean_solution()
            self.matching_nodes = self._matching_nodes()
            self.edges_in_solution = self._validate_constraints()
            print('\n########         CLEANED SOLUTION             #########\n')
            if plot:
                self.plot_solution(labels=labels)

            print(f'Similarity size: {self.similarity_size(delta):.4f}'
                f' | Similarity features: {self.similarity_features(delta):.4f}')
            print(f'There are {len(self.edges_in_solution)} conflicts in the solution')
            print(f'Following nodes were removed: {[(node,self.graph.nodes(data=True)[node]["weight"]) for node in removed_nodes]}')
