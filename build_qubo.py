from dadk.BinPol import BinPol, BitArrayShape, VarShapeSet
from modules.conflict_graph import ConflictGraph
import networkx as nx
import time


class BuildQUBO():
    def __init__(self,graph:ConflictGraph) -> None:
        self.conflict_graph = graph
        start_time = time.time()
        self._generate_variables()
        self._generate_linear_term()
        self._generate_edge_constraint()
        self.partial_time = time.time()-start_time
        self.qubo = 0
        self.time = 0
    
    @classmethod
    def from_mol_graphs(cls,mol1_graph,mol2_graph):
        conflict_graph = ConflictGraph(mol1_graph,mol2_graph)
        return cls(conflict_graph)
    
    def _hide_pol_info(self,pol):
        pol.user_data['hide_scaling_info'] = True
        pol.user_data['hide_sampling_info'] = True
        return pol
    
    def _generate_variables(self):
        # define variables
        n_vars = len(self.conflict_graph.var_list)
        bit_vars = BitArrayShape(name='node_pairs',shape=(n_vars,))
        shape_vars = VarShapeSet(bit_vars)

        # freeze var shape
        BinPol.freeze_var_shape_set(None)
    
    def _generate_linear_term(self):
        # fetch variables
        var_list = self.conflict_graph.var_list
        
        H_linear = BinPol()
        for node_pair in self.conflict_graph.graph.nodes:
            H_linear.add_term(-self.conflict_graph.graph.nodes[node_pair]['weight'],
                            #   ('node_pairs',var_list.index(node_pair)))
                              var_list.index(node_pair))

        self.H_linear = self._hide_pol_info(H_linear)

    def _generate_edge_constraint(self):
        # fetch variables
        var_list = self.conflict_graph.var_list
        
        H_constraint = BinPol()
        for edge in self.conflict_graph.graph.edges:
            edge_weight = self.conflict_graph.graph.edges[edge]['weight']

            # H_constraint.add_term(edge_weight,('node_pairs',var_list.index(edge[0])),
            #                       ('node_pairs',var_list.index(edge[1])))
            H_constraint.add_term(edge_weight,var_list.index(edge[0]),
                                  var_list.index(edge[1]))
            
        self.H_constraint = self._hide_pol_info(H_constraint)

    def create_qubo(self,factor=2.):
        start_time = time.time()
        self.qubo = self.H_linear + factor * self.H_constraint
        self.time = (time.time() - start_time) + self.partial_time
        

