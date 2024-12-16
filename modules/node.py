from modules.config import WEIGHTING_SCHEME, WEIGHTING_SCHEME_RINGS, COMPARE_METHOD, COMPARE_METHOD_RINGS, FT_NAMES
from collections import defaultdict

class Node():
    ''' Class to represent a node in the conflict graph.
    Attributes:
        - is_donor: True if the atom is a donor
        - is_acceptor: True if the atom is an acceptor
        - is_hydrophobic: True if the atom is hydrophobic
        - is_aromatic: True if the atom is aromatic
    '''
    def __init__(self,pharmacophoric,ft_factory) -> None:
        ''' Initialize the node.
        Attributes:
            - pharmacophoric: list of pharmacophoric features of the atom
            - ft_factory: feature factory of the atom
        '''
        self.is_donor = self._is_donor(ft_factory=ft_factory,pharmacophoric=pharmacophoric)
        self.is_acceptor = self._is_acceptor(ft_factory=ft_factory,pharmacophoric=pharmacophoric)
        self.is_hydrophobic = self._is_hydrophobic(ft_factory=ft_factory,pharmacophoric=pharmacophoric)
        self.is_aromatic = self._is_aromatic(ft_factory=ft_factory,pharmacophoric=pharmacophoric)

    def _is_donor(self,ft_factory,pharmacophoric):
        ''' Check if the atom is a donor.
        Attributes:
            - ft_factory: feature factory of the atom
            - pharmacophoric: list of pharmacophoric features of the atom
        '''
        return bool(FT_NAMES[ft_factory]['donor'] in pharmacophoric)
    
    def _is_acceptor(self,ft_factory,pharmacophoric):
        ''' Check if the atom is an acceptor.
        Attributes:
            - ft_factory: feature factory of the atom
            - pharmacophoric: list of pharmacophoric features of the atom
        '''
        return bool(FT_NAMES[ft_factory]['acceptor'] in pharmacophoric)

    def _is_hydrophobic(self,ft_factory,pharmacophoric):
        ''' Check if the atom is hydrophobic.
        Attributes:
            - ft_factory: feature factory of the atom
            - pharmacophoric: list of pharmacophoric features of the atom
        '''
        return bool(FT_NAMES[ft_factory]['hydrophobic'] in pharmacophoric)

    def _is_aromatic(self,ft_factory,pharmacophoric):
        ''' Check if the atom is aromatic.
        Attributes:
            - ft_factory: feature factory of the atom
            - pharmacophoric: list of pharmacophoric features of the atom
        '''
        aromatic = FT_NAMES[ft_factory]['aromatic']
        for prop in pharmacophoric:
            if prop in aromatic:
                return True
        
        return False

    def __str__(self):
        ''' String representation of the node.
        '''
        return str(self.__dict__)


class Atom(Node):
    ''' Class to represent an atom in the conflict graph.
    Attributes:
        - atomic_nb: atomic number of the atom
        - nb_implicit_h: number of implicit hydrogens
        - formal_charge: formal charge of the atom
        - partial_charge: partial charge of the atom
        - degree: degree of the atom
        - position: position of the atom in the molecule
        - bond_order: dictionary of bond orders of the atom
        - pharmacophoric: list of pharmacophoric features of the atom
        - is_donor: True if the atom is a donor
        - is_acceptor: True if the atom is an acceptor
        - is_hydrophobic: True if the atom is hydrophobic
        - is_aromatic: True if the atom is aromatic
    '''
    def __init__(self,atom, pharmacophoric, ft_factory) -> None:
        ''' Initialize the atom.
        Attributes:
            - atom: atom object from rdkit
            - pharmacophoric: list of pharmacophoric features of the atom
            - ft_factory: feature factory of the atom
        '''
        super().__init__(pharmacophoric, ft_factory)
        self._set_props(atom=atom)

    def _set_props(self,atom):
        ''' Set the properties of the atom.
        Attributes:
            - atom: atom object from rdkit
        '''
        self.atomic_nb = atom.GetAtomicNum()
        self.nb_implicit_h = atom.GetNumImplicitHs()
        self.formal_charge = atom.GetFormalCharge()
        self.partial_charge = atom.GetDoubleProp('_GasteigerCharge')
        # self.degree = atom.GetTotalDegree()
        self.degree = atom.GetDegree()
        # self.position = position 
        bond_order = defaultdict(int)
        for bond in atom.GetBonds():
            bond_order[str(bond.GetBondType())] += 1

        self.bond_order = bond_order
        # self.pharmacophoric = pharmacophoric
    
    def compare(self,other):
        ''' Compare the atom with another atom.
        Attributes:
            - other: atom to compare with
        '''
        if self.__dict__ == other.__dict__:
            # calculate total weight
            total_weight = sum(WEIGHTING_SCHEME.values())

        else:
            total_weight = 0
            for property in self.__dict__.keys():
                partial_weight = COMPARE_METHOD[property](self.__getattribute__(property),
                                                        other.__getattribute__(property))
                # print(f'property: {property}, weight: {partial_weight}')
                total_weight += WEIGHTING_SCHEME[property]*partial_weight
        return total_weight/sum(WEIGHTING_SCHEME.values())  



class Ring(Node):
    ''' Class to represent a ring in the conflict graph.
    Attributes:
        - atomic_nb: dictionary of atomic numbers of the atoms in the ring
        - nb_implicit_h: dictionary of number of implicit hydrogens of the atoms in the ring
        - formal_charge: dictionary of formal charges of the atoms in the ring
        - partial_charge: dictionary of partial charges of the atoms in the ring
        - degree: degree of the ring
        - position: position of the ring in the molecule
        - bond_order: dictionary of bond orders of the ring
        - pharmacophoric: list of pharmacophoric features of the ring
        - is_donor: True if the ring is a donor
        - is_acceptor: True if the ring is an acceptor
        - is_hydrophobic: True if the ring is hydrophobic
        - is_aromatic: True if the ring is aromatic
    '''
    def __init__(self,pharmacophoric, ft_factory,atomic_nb_dict,nb_implicit_h,
                 formal_charge,partial_charge,bonds,double) -> None:
        ''' Initialize the ring.
        Attributes:
            - pharmacophoric: list of pharmacophoric features of the ring
            - ft_factory: feature factory of the ring
            - atomic_nb_dict: dictionary of atomic numbers of the atoms in the ring
            - nb_implicit_h: dictionary of number of implicit hydrogens of the atoms in the ring
            - formal_charge: dictionary of formal charges of the atoms in the ring
            - partial_charge: dictionary of partial charges of the atoms in the ring
            - bonds: dictionary of bond orders of the ring
            - double: True if the ring is a double ring
        '''
        super().__init__(pharmacophoric, ft_factory)
        self._set_props_ring(atomic_nb_dict=atomic_nb_dict,nb_implicit_h=nb_implicit_h,
                            formal_charge=formal_charge,partial_charge=partial_charge,bonds=bonds,double=double)


    def _set_props_ring(self,atomic_nb_dict,nb_implicit_h,formal_charge,partial_charge,
                       bonds, double):
        ''' Set the properties of the ring.
        Attributes:
            - atomic_nb_dict: dictionary of atomic numbers of the atoms in the ring
            - nb_implicit_h: dictionary of number of implicit hydrogens of the atoms in the ring
            - formal_charge: dictionary of formal charges of the atoms in the ring
            - partial_charge: dictionary of partial charges of the atoms in the ring
            - bonds: dictionary of bond orders of the ring
            - double: True if the ring is a double ring
        '''

        self.atomic_nb = atomic_nb_dict
        self.nb_implicit_h = nb_implicit_h
        self.formal_charge = formal_charge
        self.partial_charge = partial_charge
        # self.degree = atom.GetTotalDegree()
        self.degree = len(bonds)
        # self.position = position
        self.double = double
        bond_order = defaultdict(int)
        for bond_atoms,bond_type in bonds.items():
            bond_order[bond_type] += 1

        self.bond_order = bond_order

    def compare(self,other):
        ''' Compare the ring with another ring.
        Attributes:
            - other: ring to compare with
        '''
        if self.__dict__ == other.__dict__:
            total_weight = sum(WEIGHTING_SCHEME_RINGS.values())
        else:
            total_weight = 0
            for property in self.__dict__.keys():
                partial_weight = COMPARE_METHOD_RINGS[property](self.__getattribute__(property),
                                                        other.__getattribute__(property))
                # print(f'property: {property}, weight: {partial_weight}')
                total_weight += WEIGHTING_SCHEME_RINGS[property]*partial_weight
        return total_weight/sum(WEIGHTING_SCHEME_RINGS.values())  