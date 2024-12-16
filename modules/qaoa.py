from modules.conflict_graph import ConflictGraph
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from scipy.optimize import minimize


class QAOA():
    ''' Class to represent the QAOA algorithm.
    Attributes:
        - graph: conflict graph
        - var_list: list of variables in the conflict graph
        - n_qubits: number of qubits in the circuit
        - factor: factor to multiply the constraint by
    '''
    def __init__(self,conflict_g:ConflictGraph,n_layers,factor=2.) -> None:
        self.graph = conflict_g.graph
        self.var_list = conflict_g.var_list
        self.n_qubits = self.graph.number_of_nodes()
        self.factor = factor
        self.max_weight = max([conflict_g.graph.nodes()[pair]['weight'] for pair in conflict_g.graph.nodes()])
        self.n_layers = n_layers
        self.ntheta = n_layers*(1+self.n_qubits)
        self.pqc = self.create_qaoa_circ()

    def create_qaoa_circ (self):
        #global Theta
        p = self.ntheta//(1+self.n_qubits)
        qc = QuantumCircuit (self.n_qubits)

        Theta = ParameterVector('θ', self.ntheta)
        
        gamma = Theta [:p] # as many gamma as layers
        beta = Theta [p:] # n_qubits beta parameters per layer
      
        # estado inicial
        for i in range (0, self.n_qubits):
            qc.h(i)
        ip = 0
        for irep in range (0, p):

            # Hamiltoniano problema = Hamiltoniano Obxectivo + Hamiltoniano Penalizacion (Constraint)

            #H_O
            for i,node in enumerate(self.graph.nodes(data=True)):
                qc.rz (node[1]['weight'] / self.max_weight * gamma [irep], i)
                
            #H_C
            for pair in list (self.graph.edges()):
                node1 = self.graph.nodes()[pair[0]]
                node2 = self.graph.nodes()[pair[1]]
                qubit0 = self.var_list.index(pair[0])
                qubit1 = self.var_list.index(pair[1])
                qc.rz (-0.5 * self.factor * min(node1['weight'],node2['weight']) / self.max_weight * gamma [irep] , qubit0)
                qc.rz (-0.5 * self.factor * min(node1['weight'],node2['weight']) / self.max_weight * gamma [irep] , qubit1)
                qc.rzz (0.5 * self.factor * min(node1['weight'],node2['weight']) / self.max_weight * gamma [irep] , qubit0,qubit1)
                #qc.barrier()    
            
            #Hamiltoniano mestura
            for i in range (self.n_qubits):
                qc.rx (2. * beta [ip], i)
                ip += 1     
        
        qc.measure_all()
        
        return qc



    def get_expectation (self, theta, backend, n_shots):

        qc = self.pqc.assign_parameters(theta)
        
        counts = backend.run (qc, seed_simulator = 10, shots = n_shots).result().get_counts()
       
        avg = 0
        sum_count = 0
        for x, count in counts.items():

            x = x[::-1] # invert order bc qiskit returns little endian ordering
            obj = 0
            for i, node in enumerate(self.graph.nodes(data=True)):
                if x [i] == '1':
                    obj -= node[1]['weight'] / self.max_weight

                if x [i] == '0':
                    obj += node[1]['weight'] / self.max_weight

            constraint = 0
            for pair in self.graph.edges():
                qubit0 = self.var_list.index(pair[0])
                qubit1 = self.var_list.index(pair[1])

                if  x [qubit0] == '1' and x [qubit1] == '1':
                    node1 = self.graph.nodes()[pair[0]]
                    node2 = self.graph.nodes()[pair[1]]
                    constraint += min(node1['weight'], node2['weight']) / self.max_weight

                if  x [qubit0] == '0' or x [qubit1] == '0':
                    node1 = self.graph.nodes()[pair[0]]
                    node2 = self.graph.nodes()[pair[1]]
                    constraint -= min(node1['weight'], node2['weight']) / self.max_weight  

            obj = obj + self.factor * constraint      

            #obj = mwis_obj (bitstring, G, w, weightH_C) 
            avg += obj * count
            sum_count += count

        return avg / sum_count     

    