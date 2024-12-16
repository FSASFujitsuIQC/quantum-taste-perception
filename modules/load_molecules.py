from rdkit import Chem
import os


class LoadMolecule():
    ''' Class to load molecules in .mol2 or .sdf format.
    '''
    def __init__(self,par_dir='Datos/terpenoids') -> None:
        self.par_dir = par_dir
        self.available_molecules_mol2, self.available_molecules_sdf = self._available_molecules()
        self.available_molecules = set(self.available_molecules_mol2).union(
            set(self.available_molecules_sdf))

    def _available_molecules(self):
        ''' Append available molecules in .mol2 or .sdf format
        '''
        diff_mol_mol2 = []
        diff_mol_sdf = []
        for f in os.listdir(self.par_dir):
            if f[-5:] == '.mol2':
                if f[:-5] not in diff_mol_mol2:
                    diff_mol_mol2.append(f[:-5])
            elif f[-4:] == '.sdf':
                if f[:-4] not in diff_mol_sdf:
                    diff_mol_sdf.append(f[:-4])

        print(f'{len(set(diff_mol_mol2).union((diff_mol_sdf)))} molecules available in {self.par_dir}')
        
        # return {'mol2':diff_mol_mol2,'sdf':diff_mol_sdf}
        return diff_mol_mol2,diff_mol_sdf
    
    def load(self,mol_filename,removeHs=False):
        '''Load mol_filename molecule'''
        
        if mol_filename in self.available_molecules_mol2:
            filename = self.par_dir + '/' + mol_filename + '.mol2'
            mol = Chem.MolFromMol2File(filename,removeHs=removeHs)
            if mol == None:
                pass
            else:
                return mol
        
        if mol_filename in self.available_molecules_sdf:
            filename = self.par_dir + '/' + mol_filename + '.sdf'
            inf = open(filename,'rb')
            with Chem.ForwardSDMolSupplier(inf,removeHs=removeHs) as fsuppl:
                for mol in fsuppl:
                    if mol is None: 
                        raise ValueError('Molecule file is empty')
                    return mol
        else:
            raise ValueError(f'Molecule {mol_filename} is not available in'
                             f'.sdf or .mol2 in folder {self.par_dir}')
        
    def load_all_available(self):
        '''Load all available molecules'''
        mol_list = []
        for mol_filename in self.available_molecules:
            mol_list.append(self.load(mol_filename=mol_filename))
        return mol_list
