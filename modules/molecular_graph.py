import os
import time
import numpy as np
from collections import defaultdict
import rdkit
from rdkit import Chem, RDConfig
from rdkit.Chem import AllChem, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from modules.node import Atom, Ring
import networkx as nx
from modules.config import fdef_string


class MolecularGraph():
    '''
    Build molecular graph from rdkit molecule object. Arguments:
    - molecule: rdkit molecule object
    - mol_name: name of molecule
    - collapse_rings: if True, rings are collapsed into a single node
    - ft_factory: pharmacophoric feature factory. Options: 'Gobbi' or 'base'
    '''
    def __init__(self,molecule:rdkit.Chem.rdchem.Mol,mol_name='',
                 collapse_rings=False,ft_factory = 'Gobbi') -> None:
        self.name = mol_name
        self.mol = molecule
        start_time = time.time()
        self.positions = self._get_positions()
        self.pharmacophoric_per_atom = self._get_pharmacophoric_features_per_atom(ft_factory)
        self.distance_matrix = self._get_distance_matrix()
        self._compute_partial_charges()
        self.mol_graph = self.build_graph(ft_factory,collapse_rings)
            
        self.time = time.time()-start_time

    def _get_distance_matrix(self):
        ''' Compute distance matrix of molecule's atoms'''
        return Chem.Get3DDistanceMatrix(self.mol)
    
    def _get_positions(self):
        ''' Compute 3D positions of molecule's atoms'''
        embed = False
        if self.mol.GetNumConformers() == 0:
            embed = True
        elif not self.mol.GetConformer().Is3D():
            embed = True
        if embed:
            self.mol = AllChem.AddHs(self.mol)
            rdDistGeom.EmbedMolecule(self.mol)
        # self.mol = AllChem.RemoveHs(self.mol)
        return self.mol.GetConformer().GetPositions()
    
    def _compute_partial_charges(self):
        ''' Compute partial charges of molecule's atoms'''
        ComputeGasteigerCharges(self.mol)
    
    def _get_pharmacophoric_features_per_atom(self,ft_factory='Gobbi',RemoveHs=True):
        ''' Compute pharmacophoric features of molecule's atoms'''

        if ft_factory == 'Gobbi':
            fdef = fdef_string
            feature_factory = AllChem.BuildFeatureFactoryFromString(fdef)

        elif ft_factory == 'base':
            # load feature definition file
            fdef = os.path.join(RDConfig.RDDataDir,'BaseFeatures.fdef')
            feature_factory = AllChem.BuildFeatureFactory(fdef)

        else:
            raise ValueError(f'Unknown feature factory method: {ft_factory}')

        # get features
        features = feature_factory.GetFeaturesForMol(self.mol)

        features_per_atom = {atom:[] for atom in range(self.mol.GetNumAtoms())}
        for feature in features:
            for atom in feature.GetAtomIds():
                features_per_atom[atom].append(feature.GetType())
        
        if RemoveHs:
            self.mol = AllChem.RemoveHs(self.mol)

        return features_per_atom
    
    def build_graph(self,ft_factory,collapse_rings=False):
        # instantiate graph
        G = nx.Graph()

        # if we collapse rings, we need r_info for both add_atoms and add_rings
        r_info = None
        if collapse_rings:
            AllChem.FastFindRings(self.mol)
            r_info = self.mol.GetRingInfo()

        # add atoms and edges between them
        G = self._add_atoms(G,ft_factory=ft_factory,only_non_rings=collapse_rings,r_info=r_info)

        if collapse_rings:
            # add collapsed rings and remaining edges
            G = self._add_rings(G,r_info=r_info,ft_factory=ft_factory)
        
        return G
        
    def _add_atoms(self,G,ft_factory,only_non_rings=False,**kargs):
        ''' Build molecular graph. Arguments:
        - ft_factory: pharmacophoric feature factory. Options: 'Gobbi' or 'base'
        - only_non_rings: if True, only atoms not in rings are added to the graph
        - kargs: additional arguments to be passed to _get_ring_props method
        '''
        pharmacophoric = self.pharmacophoric_per_atom
        distances = self.distance_matrix
        self.pos = {}
        
        # add nodes 
        for atom in self.mol.GetAtoms():
            if atom.IsInRing() and only_non_rings:
                continue

            atom_idx = atom.GetIdx()
            node = Atom(atom=atom,pharmacophoric=pharmacophoric[atom_idx],
                        ft_factory=ft_factory)
            
            G.add_node(atom_idx,features=node)
            self.pos[atom_idx] = self.positions[atom_idx]

            # add edges
            for bond in atom.GetBonds():
                neighbour_idx = bond.GetOtherAtomIdx(atom_idx)
                if only_non_rings and len(kargs['r_info'].AtomMembers(neighbour_idx)) > 0: # this means that neighbour is in ring
                    continue

                if G.has_node(neighbour_idx):
                    G.add_edge(atom_idx,neighbour_idx,
                               distance=distances[atom_idx,neighbour_idx],
                               bond_type=str(bond.GetBondType()))
            
        return G

    def _get_ring_props(self,ring_id,ring,r_info,inside_bonds):
        '''
        Get properties of ring. Arguments:
        - ring_id: ring id
        - ring: ring atoms ids
        - r_info: ring info object
        - inside_bonds: bonds inside ring
        '''
        atomic_nb_dict = defaultdict(int)
        nb_implicit_h = 0
        formal_charge = 0
        partial_charge = 0
        bonds = {} # dict, key= nodes ids; value= bond type as str
        positions_list = []
        double = False

        for atom_idx in ring:
            # group properties from atoms in ring
            atom = self.mol.GetAtomWithIdx(atom_idx)
            positions_list.append(self.positions[atom_idx])
            atomic_nb_dict[atom.GetAtomicNum()] += 1
            nb_implicit_h += atom.GetNumImplicitHs()
            formal_charge += atom.GetFormalCharge()
            partial_charge += atom.GetDoubleProp('_GasteigerCharge')

            # check bonds & their types 
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                # only bonds that go outside current ring
                if not r_info.AreAtomsInSameRing(atom_idx,neighbor_idx):
                    bond = self.mol.GetBondBetweenAtoms(atom_idx,neighbor_idx)
                    # if other atom is in ring, different var name
                    if neighbor.IsInRing():
                        for other_ring in r_info.AtomMembers(neighbor_idx):
                            bonds[('r'+str(ring_id),'r'+str(other_ring))] = str(bond.GetBondType())
                    else:
                        bonds[('r'+str(ring_id),neighbor_idx)] = str(bond.GetBondType())
            # case atom shared between rings, artificial bond
            if r_info.NumAtomRings(atom_idx) > 1:
                for other_ring in r_info.AtomMembers(atom_idx):
                    if other_ring != ring_id:
                        bonds[('r'+str(ring_id),'r'+str(other_ring))] = 'artificial'
        
        # check if there are double bonds within ring
        inside_bonds_types = [str(self.mol.GetBondWithIdx(i).GetBondType()) for i in inside_bonds[ring_id]]
        if 'DOUBLE' in inside_bonds_types: double = True

        return  positions_list, atomic_nb_dict, nb_implicit_h, formal_charge, partial_charge, bonds, double    

    def _add_rings(self,G,r_info,ft_factory):
        '''
        Build molecular graph merging atoms in rings. Arguments:
        - ft_factory: pharmacophoric feature factory. Options: 'Gobbi' or 'base'
        '''
        # Fetch ring & other relevant info
        rings = r_info.AtomRings()
        inside_bonds = r_info.BondRings()
        pharmacophoric = self.pharmacophoric_per_atom
        ring_positions = np.zeros((len(rings),3))
        self.rings = {}

        # add ring artificial nodes
        for ring_id,ring in enumerate(rings):
            # get props
            (positions_list, atomic_nb_dict, nb_implicit_h, formal_charge, 
             partial_charge, bonds, double)  = self._get_ring_props(ring_id=ring_id,ring=ring,r_info=r_info,
                                                            inside_bonds=inside_bonds)
            
            # ring position: ring centroid
            ring_positions[ring_id] = np.mean(positions_list,axis=0)
            self.pos['r'+str(ring_id)] = ring_positions[ring_id]

            # node object storing all ring's proporties & add node
            node = Ring(pharmacophoric=[prop for idx in ring for prop in pharmacophoric[idx]],
                                       ft_factory=ft_factory,atomic_nb_dict=atomic_nb_dict,
                                       nb_implicit_h=nb_implicit_h,formal_charge=formal_charge,
                                       partial_charge=partial_charge,bonds=bonds,double=double)
            G.add_node('r'+str(ring_id),features=node)
            self.rings['r'+str(ring_id)] = ring

            # debugging
            # print(bonds)

            # add remaining edges (involving rings)
            for bond_elements,bond_type in bonds.items():
                if G.has_node(bond_elements[1]):
                    # print(type(bond_elements[1]))
                    if type(bond_elements[1]) == str and bond_elements[1][0]=='r': # this means variable is a ring 
                        other_ring_id = int(bond_elements[1][1])
                        distance = np.linalg.norm(ring_positions[other_ring_id]-ring_positions[ring_id])
                    else:
                        # print(bond_elements[1],ring_id)
                        distance = np.linalg.norm(self.positions[bond_elements[1]]-ring_positions[ring_id])

                    G.add_edge(bond_elements[0],bond_elements[1],
                               distance=distance,
                               bond_type=bond_type)
        return G
    
    def get_total_features(self):
        '''
        Get total number of features in molecule. This is used to compute the metric similitaruty
        features.
        '''
        # total_features = 0
        # for node in self.mol_graph.nodes(data=True):
        #     node_ft = node[1]['features']
        #     if type(node_ft) == Ring:
        #         weight = sum(list(node_ft.atomic_nb.values()))
        #     else:
        #         weight = 1
        #     total_features += weight
        total_features = self.mol.GetNumHeavyAtoms()

        return total_features